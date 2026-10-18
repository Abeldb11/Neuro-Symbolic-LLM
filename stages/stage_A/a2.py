"""Stage A2: the PC-inspired auxiliary objective on top of the A1 residual adapter.

    L_A2 = L_task + lambda_kl * KL(p_F0 || p_F0+R) + lambda_wd * ||phi||^2   (A1, unchanged)
         + lambda_cp   * L_cpc                                               (contrastive future)
         + lambda_stab * (w_anch * L_zanch + w_hep * L_hep-nudge)            (stability)

Backprop only: there is no settling in A2, so "PC-inspired" means the auxiliary
*losses* are PC-flavoured, not that the weights are trained by predictive coding.
Where FabricPC applies (the CPC heads and the squared-error energies of the anchor
and HEP terms) it supplies the nodes; the pure math lives in ``residual/``.

Everything stage-level is in this one file, in this order:

1. configuration   ``CPCConfig``, ``HEPConfig``, ``StabilityConfig``, ``A2Config``
2. parameters      ``init_a2_params`` and friends (one flat, two-level tree)
3. terms           ``cpc_term``, ``stability_term``: pure functions of hidden-state
                   dictionaries, so they never touch the substrate
4. objective       ``a2_objective``: the A1 ``Objective`` contract for ``run_sequential``

Not here yet (teammates' PRs): the hidden-prediction loss ``L_hid`` (``P_l``),
``L_step`` and ``L_res-smooth``. Each adds a config section, a ``*_term`` function
and a few lines in ``a2_objective``; the runner and the parameter tree need no change.

Parameter tree (``run_sequential`` differentiates all of it with one optimiser)::

    layer_<l>      {A, B}        adapter                  (A1)
    cpcq_<l>       {W}           CPC projection q_l       (lambda_cp > 0)
    cpca_<l>_<k>   {M}           CPC predictor A_{l,k}    (lambda_cp > 0)
    hep_<l>        {V} | {U, W}  trainable highway        (lambda_stab > 0, hep.kind dense / lowrank)

``ResidualAdapter.num_params`` and ``l2_norm_sq`` must only see the ``layer_<l>``
sub-tree (``adapter_params``): the weight decay of A1 is on the adapter alone.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp

from frozenllm.substrate import FrozenSubstrate, identity_modify
from residual import AdapterParams, ResidualAdapter
from residual.a2_losses import (
    anchor_loss,
    contrastive_future_loss,
    hep_nudge_loss,
    output_error,
)
from residual.cpc_heads import BACKENDS as CPC_BACKENDS
from residual.cpc_heads import (
    CPCParams,
    a_key,
    init_cpc_heads,
    q_key,
    split_cpc,
)
from residual.fpc_energy import BACKENDS as ENERGY_BACKENDS
from residual.hep import (
    KINDS,
    TRAINABLE_KINDS,
    HEPParams,
    highway_for,
    init_hep,
    split_hep,
)
from stages.stage_base import base_logits, checked_kwargs

# =============================================================================
# 1. Configuration
# =============================================================================


POSITIVES = ("base", "corrected")


@dataclass(frozen=True)
class CPCConfig:
    """``objective.cpc`` section of the A2 YAML."""

    # lambda_cp; 0 switches the term (and its heads) off.
    weight: float = 0.0
    # Future offsets k (the paper: "such as 1, 2 or 4"; start with (1,)): the
    # positive is the code k positions ahead. One predictor A_{l,k} per offset.
    offsets: tuple[int, ...] = (1,)
    # m: width of the CPC code.
    dim: int = 32
    tau: float = 0.1
    # "base": positive from the unmodified model h^base; "corrected": from h~.
    positive: str = "base"
    # Deterministic stride subsampling of anchor positions; null keeps all.
    max_anchors_per_seq: int | None = 256
    # Std of the noise added to the identity predictor A_{l,k}.
    a_init_noise: float = 0.01
    # "fabricpc": the heads q and A run as FabricPC nodes; "jax": pure-JAX reference.
    heads: str = "fabricpc"

    def __post_init__(self) -> None:
        if not isinstance(self.weight, int | float) or self.weight < 0:
            raise ValueError(f"weight must be a number >= 0, got {self.weight!r}")
        offsets = self.offsets
        if isinstance(offsets, int) and not isinstance(offsets, bool):
            offsets = (offsets,)
        offsets = tuple(offsets) if isinstance(offsets, list | tuple) else offsets
        if (
            not isinstance(offsets, tuple)
            or not offsets
            or any(
                not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in offsets
            )
            or len(set(offsets)) != len(offsets)
        ):
            raise ValueError(
                f"offsets must be distinct integers >= 1, got {self.offsets!r}"
            )
        object.__setattr__(self, "offsets", offsets)
        if self.heads not in CPC_BACKENDS:
            raise ValueError(
                f"heads must be one of {list(CPC_BACKENDS)}, got {self.heads!r}"
            )
        if not isinstance(self.dim, int) or isinstance(self.dim, bool) or self.dim < 1:
            raise ValueError(f"dim must be an integer >= 1, got {self.dim!r}")
        if not self.tau > 0:
            raise ValueError(f"tau must be > 0, got {self.tau!r}")
        if self.positive not in POSITIVES:
            raise ValueError(
                f"positive must be one of {list(POSITIVES)}, got {self.positive!r}"
            )
        if self.max_anchors_per_seq is not None and (
            not isinstance(self.max_anchors_per_seq, int)
            or isinstance(self.max_anchors_per_seq, bool)
            or self.max_anchors_per_seq < 1
        ):
            raise ValueError(
                "max_anchors_per_seq must be an integer >= 1 or null, "
                f"got {self.max_anchors_per_seq!r}"
            )
        if not self.a_init_noise >= 0:
            raise ValueError(f"a_init_noise must be >= 0, got {self.a_init_noise!r}")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CPCConfig:
        data = dict(data)
        if "offset" in data:  # single-offset spelling
            if "offsets" in data:
                raise ValueError("give either 'offset' or 'offsets', not both")
            data["offsets"] = (data.pop("offset"),)
        return cls(**checked_kwargs(cls, data))

    @property
    def enabled(self) -> bool:
        return self.weight > 0

    @property
    def needs_base_states(self) -> bool:
        return self.enabled and self.positive == "base"


NORMALIZATIONS = ("relative", "none")


def _number(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or value < 0:
        raise ValueError(f"{name} must be a number >= 0, got {value!r}")


@dataclass(frozen=True)
class HEPConfig:
    """``objective.stab.hep`` section of the A2 YAML."""

    # null switches HEP off (no nudge metric, no HEP loss).
    kind: str | None = None
    # Rank of the low-rank / fixed-random highway.
    rank: int = 16
    # alpha_y: gain of the nudge (the program sweeps it).
    alpha: float = 1.0

    def __post_init__(self) -> None:
        if self.kind is not None and self.kind not in KINDS:
            raise ValueError(
                f"kind must be null or one of {list(KINDS)}, got {self.kind!r}"
            )
        if (
            not isinstance(self.rank, int)
            or isinstance(self.rank, bool)
            or self.rank < 1
        ):
            raise ValueError(f"rank must be an integer >= 1, got {self.rank!r}")
        _number("alpha", self.alpha)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HEPConfig:
        return cls(**checked_kwargs(cls, data))

    @property
    def trainable(self) -> bool:
        return self.kind in TRAINABLE_KINDS


@dataclass(frozen=True)
class StabilityConfig:
    """``objective.stab`` section of the A2 YAML."""

    # lambda_stab; 0 switches the whole term off.
    weight: float = 0.0
    w_anch: float = 1.0
    # 0: HEP nudge is only logged. > 0 needs a trainable highway (dense / lowrank).
    w_hep: float = 0.0
    normalize: str = "relative"
    # "fabricpc": the squared errors are FabricPC node energies (IdentityNode for the
    # anchor, Linear node(s) for the highway); "jax": plain arithmetic (test oracle).
    backend: str = "fabricpc"
    hep: HEPConfig = field(default_factory=HEPConfig)

    def __post_init__(self) -> None:
        for name in ("weight", "w_anch", "w_hep"):
            _number(name, getattr(self, name))
        if self.normalize not in NORMALIZATIONS:
            raise ValueError(
                f"normalize must be one of {list(NORMALIZATIONS)}, got {self.normalize!r}"
            )
        if self.backend not in ENERGY_BACKENDS:
            raise ValueError(
                f"backend must be one of {list(ENERGY_BACKENDS)}, got {self.backend!r}"
            )
        if self.w_hep > 0 and self.hep.kind is None:
            raise ValueError("w_hep > 0 needs hep.kind (dense or lowrank).")
        if self.w_hep > 0 and not self.hep.trainable:
            raise ValueError(
                f"w_hep > 0 with hep.kind={self.hep.kind!r}: the highway is fixed and "
                "eps is under stop-gradient, so the term has no gradient. Set w_hep = 0 "
                "(the nudge norm is still logged) or use kind dense / lowrank."
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StabilityConfig:
        raw = dict(checked_kwargs(cls, data))
        raw["hep"] = HEPConfig.from_dict(raw.get("hep") or {})
        return cls(**raw)

    @property
    def enabled(self) -> bool:
        return self.weight > 0

    @property
    def needs_base_states(self) -> bool:
        return self.enabled and self.w_anch > 0

    @property
    def needs_output_error(self) -> bool:
        return self.enabled and self.hep.kind is not None


@dataclass(frozen=True)
class A2Config:
    """``objective:`` section of the A2 YAML (nested ``cpc`` and ``stab`` sections)."""

    # A1 terms, same names and meaning as A1Config.
    lambda_kl: float = 0.0
    lambda_wd: float = 0.0
    # Seed of the auxiliary heads and of a fixed-random highway (the adapter has its own).
    seed: int = 0
    cpc: CPCConfig = field(default_factory=CPCConfig)
    stab: StabilityConfig = field(default_factory=StabilityConfig)

    def __post_init__(self) -> None:
        _number("lambda_kl", self.lambda_kl)
        _number("lambda_wd", self.lambda_wd)
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f"seed must be an integer, got {self.seed!r}")
        if self.stab.hep.kind == "tied":
            raise ValueError(
                "hep.kind='tied' is not wired into a2_objective yet (it needs the frozen "
                "LM-head matrix); use dense, lowrank or fixed_random."
            )

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> A2Config:
        raw = dict(checked_kwargs(cls, data))
        raw["cpc"] = CPCConfig.from_dict(raw.get("cpc") or {})
        raw["stab"] = StabilityConfig.from_dict(raw.get("stab") or {})
        return cls(**raw)

    @property
    def needs_base_states(self) -> bool:
        """Whether the objective needs the base model's hidden states ``h^base``."""
        return self.cpc.needs_base_states or self.stab.needs_base_states

    @property
    def aux_enabled(self) -> bool:
        return self.cpc.enabled or self.stab.enabled


# =============================================================================
# 2. Parameters
# =============================================================================

# The flat tree is two levels deep (``RunWriter.save_params`` requires it).
ADAPTER_PREFIX = "layer_"  # layer_key(l) == f"{ADAPTER_PREFIX}{l}"


def adapter_params(params: Mapping[str, Mapping[str, jax.Array]]) -> AdapterParams:
    """The ``layer_<l>`` entries: what ``ResidualAdapter`` and the A1 weight decay see."""
    return {k: dict(v) for k, v in params.items() if k.startswith(ADAPTER_PREFIX)}


def init_a2_params(config: A2Config, adapter: ResidualAdapter) -> dict[str, Any]:
    """The adapter parameters plus the heads the enabled terms need.

    With every auxiliary weight at 0 this is exactly ``adapter.init_params()``.
    """
    d = adapter.hidden_size
    layers = tuple(adapter.layers)
    params: dict[str, Any] = dict(adapter.init_params())
    heads = {
        **init_cpc_params(config.cpc, config.seed, layers, d),
        **init_hep_params(
            config.stab, config.seed, layers, adapter.architecture.vocab_size, d
        ),
    }
    clash = sorted(set(heads) & set(params))
    if clash:
        raise ValueError(f"head parameters clash with the adapter keys: {clash}")
    params.update(heads)
    return params


def param_counts(
    params: Mapping[str, Mapping[str, jax.Array]],
) -> dict[str, int]:
    """Trainable parameters by group: ``adapter``, ``cpc``, ``hep`` and ``total``."""

    def count(tree: Mapping[str, Mapping[str, jax.Array]]) -> int:
        return sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(tree))

    groups = {
        "adapter": count(adapter_params(params)),
        "cpc": count(split_cpc(params)),
        "hep": count(split_hep(params)),
    }
    other = count(params) - sum(groups.values())
    if other:
        raise ValueError(f"{other} parameters belong to no known group.")
    return {**groups, "total": sum(groups.values())}


def init_cpc_params(
    config: CPCConfig, seed: int, layers: tuple[int, ...], hidden_size: int
) -> CPCParams:
    """Heads for ``layers``; empty when the term is disabled."""
    if not config.enabled:
        return {}
    return init_cpc_heads(
        seed,
        layers,
        hidden_size,
        config.dim,
        config.offsets,
        config.a_init_noise,
    )


def init_hep_params(
    config: StabilityConfig,
    seed: int,
    layers: tuple[int, ...],
    vocab_size: int,
    hidden_size: int,
) -> HEPParams:
    """Trainable highway parameters; empty unless HEP is enabled with a trainable kind."""
    if not (config.enabled and config.hep.trainable):
        return {}
    assert config.hep.kind is not None
    return init_hep(
        config.hep.kind, seed, layers, vocab_size, hidden_size, config.hep.rank
    )


# =============================================================================
# 3. Terms (pure functions of hidden-state dictionaries ``{layer: [B, T, d]}``)
# =============================================================================


def token_mask(labels: jax.Array | None, ignore_index: int = -100) -> jax.Array | None:
    """``[B, T]`` validity of the tokens from the labels: ``labels != ignore_index``.

    ``None`` when there are no labels (every token is then treated as real). Note
    that tokens whose label is ignored for the task loss (a prompt, say) are also
    left out of the contrastive term; pass an explicit ``mask`` to ``cpc_term``
    to keep them.
    """
    if labels is None:
        return None
    return labels != ignore_index


def cpc_term(
    params: Mapping[str, Mapping[str, jax.Array]],
    *,
    config: CPCConfig,
    corrected: Mapping[int, jax.Array],
    base: Mapping[int, jax.Array] | None = None,
    mask: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Unweighted ``L_cpc`` (mean over sites ``(l, k)``) and its diagnostics.

    Args:
        params: a tree holding the heads ``cpcq_<l>`` / ``cpca_<l>_<k>``.
        corrected: ``{layer: h~_l}``, each ``[..., B, T, d]``, carrying gradient.
        base: ``{layer: h^base_l}``; required when ``config.positive == "base"``.
        mask: optional ``[B, T]`` token validity (``False`` = padding); see
            :func:`token_mask`. Positions outside it are never anchors, positives
            or negatives.
    """
    if not corrected:
        raise ValueError("cpc_term needs at least one site.")
    if config.positive == "base" and base is None:
        raise ValueError("positive='base' requires the base hidden states.")
    losses = []
    metrics: dict[str, list[jax.Array]] = {}
    for layer, anchors in corrected.items():
        if config.positive == "base":
            assert base is not None
            if layer not in base:
                raise KeyError(f"missing base hidden state for layer {layer}.")
            targets = base[layer]
        else:
            targets = anchors
        for offset in config.offsets:
            loss, m = contrastive_future_loss(
                params[q_key(layer)],
                params[a_key(layer, offset)],
                anchors,
                targets,
                offset=offset,
                tau=config.tau,
                max_anchors_per_seq=config.max_anchors_per_seq,
                mask=mask,
                backend=config.heads,
            )
            losses.append(loss)
            for name, value in m.items():
                metrics.setdefault(name, []).append(value)
    reduced = {name: jnp.mean(jnp.stack(vals)) for name, vals in metrics.items()}
    return jnp.mean(jnp.stack(losses)), reduced


def stability_term(
    params: Mapping[str, Mapping[str, jax.Array]],
    *,
    config: StabilityConfig,
    corrected: Mapping[int, jax.Array],
    base: Mapping[int, jax.Array] | None = None,
    logits: jax.Array | None = None,
    labels: jax.Array | None = None,
    seed: int = 0,
    tied: jax.Array | None = None,
    mask: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """``w_anch * L_zanch + w_hep * L_hep-nudge`` (without ``lambda_stab``) and diagnostics.

    Args:
        params: a tree holding the trainable highway entries ``hep_<l>`` (if any).
        corrected: ``{layer: h~_l}``, each ``[..., B, T, d]``, carrying gradient.
        base: ``{layer: h^base_l}``; required when ``w_anch > 0``.
        logits, labels: adapted logits ``[B, T, V]`` and the labels (the inputs when
            training on plain text); required when HEP is configured. ``-100`` labels
            are ignored.
        seed: seed of the ``fixed_random`` highway.
        tied: ``[V, d]`` matrix for ``hep.kind == "tied"``.
        mask: optional ``[B, T]`` token validity (``False`` = padding) for the anchor
            term, e.g. ``token_mask(labels)``. The HEP term masks by the
            ignored labels on its own.
    """
    if not corrected:
        raise ValueError("stability_term needs at least one site.")
    total = jnp.zeros((), jnp.float32)
    metrics: dict[str, jax.Array] = {}

    if config.w_anch > 0:
        if base is None:
            raise ValueError("w_anch > 0 requires the base hidden states.")
        losses, parts = [], []
        for layer, h in corrected.items():
            if layer not in base:
                raise KeyError(f"missing base hidden state for layer {layer}.")
            loss, m = anchor_loss(
                h,
                base[layer],
                normalize=config.normalize,
                mask=mask,
                backend=config.backend,
            )
            losses.append(loss)
            parts.append(m)
        l_anch = jnp.mean(jnp.stack(losses))
        total = total + config.w_anch * l_anch
        metrics["stab_anch"] = l_anch
        for name in parts[0]:
            metrics[name] = jnp.mean(jnp.stack([m[name] for m in parts]))

    if config.hep.kind is not None:
        if logits is None or labels is None:
            raise ValueError("HEP needs the adapted logits and the labels.")
        eps, eps_mask = output_error(logits, labels)
        n_pos = eps.shape[-2]
        losses, parts = [], []
        for layer, h in corrected.items():
            v = highway_for(
                config.hep.kind,
                layer,
                params=params,
                seed=seed,
                d_out=eps.shape[-1],
                d_hidden=h.shape[-1],
                rank=config.hep.rank,
                tied=tied,
            )
            # eps at position t is the error of the prediction made at t.
            loss, m = hep_nudge_loss(
                eps,
                v,
                config.hep.alpha,
                reference=h[..., :n_pos, :],
                mask=eps_mask,
                backend=config.backend,
            )
            losses.append(loss)
            parts.append(m)
        l_hep = jnp.mean(jnp.stack(losses))
        if config.w_hep > 0:
            total = total + config.w_hep * l_hep
        metrics["stab_hep"] = l_hep
        for name in parts[0]:
            metrics[name] = jnp.mean(jnp.stack([m[name] for m in parts]))

    return total, metrics


# =============================================================================
# 4. Objective
# =============================================================================


def a2_objective(
    params: Mapping[str, Mapping[str, jax.Array]],
    *,
    substrate: FrozenSubstrate,
    adapter: ResidualAdapter,
    config: A2Config,
    input_ids: jax.Array,
    labels: jax.Array | None = None,
    base: jax.Array | None = None,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """``(total, terms)`` of the A2 objective; the ``Objective`` contract of ``run_sequential``.

    ``params`` is the whole flat tree of :func:`init_a2_params`. ``base`` is optional
    precomputed base *logits* (as in A1). ``terms`` holds ``total``, ``task``,
    ``wd`` (and ``kl`` when computed), plus ``cpc`` / ``stab`` and their diagnostics
    for the enabled terms; ``cpc`` and ``stab`` are logged *before* their lambda.

    With ``cpc.weight == stab.weight == 0`` the result equals ``a1_objective``.
    """
    ap = adapter_params(params)
    run = substrate.run_with_interception(
        input_ids, modify_fn=adapter.modify_fn(ap), intercept_layers=adapter.layers
    )
    adapted = run.logits
    target_labels = input_ids if labels is None else labels
    wd = ResidualAdapter.l2_norm_sq(ap)

    need_states = config.needs_base_states
    need_kl = config.lambda_kl > 0 or base is not None
    base_states: dict[int, jax.Array] | None = None
    base_lg: jax.Array | None = None
    if need_states:
        # One extra forward gives both the base states and (if not given) base logits.
        base_run = substrate.run_with_interception(
            input_ids, modify_fn=identity_modify, intercept_layers=adapter.layers
        )
        base_states = {
            layer: jax.lax.stop_gradient(base_run.hidden_state(layer))
            for layer in adapter.layers
        }
        base_lg = jax.lax.stop_gradient(base_run.logits if base is None else base)
    elif need_kl:
        base_lg = (
            base_logits(substrate, input_ids)
            if base is None
            else jax.lax.stop_gradient(base)
        )

    terms: dict[str, jax.Array]
    if base_lg is None:  # no KL: skip it rather than log NaN (strict JSON rejects it)
        task = substrate.compute_loss(adapted, target_labels, losses=("CE",))
        total = task
        terms = {"task": task}
    else:
        total, components = substrate.compute_loss(
            adapted,
            target_labels,
            losses=("CE", "KL"),
            base_logits=base_lg,
            weights={"KL": config.lambda_kl},
            return_components=True,
        )
        task = components["ce"]
        terms = {"task": task, "kl": components["kl"]}
    total = total + config.lambda_wd * wd
    terms["wd"] = wd

    if config.aux_enabled:
        # h~_l = R applied to the pristine block output (what the next block receives).
        corrected = {
            layer: adapter.apply(ap, run.hidden_state(layer), layer)
            for layer in adapter.layers
        }
        mask = token_mask(labels)
        if config.cpc.enabled:
            l_cpc, m = cpc_term(
                params,
                config=config.cpc,
                corrected=corrected,
                base=base_states,
                mask=mask,
            )
            total = total + config.cpc.weight * l_cpc
            terms["cpc"] = l_cpc
            terms.update(m)
        if config.stab.enabled:
            l_stab, m = stability_term(
                params,
                config=config.stab,
                corrected=corrected,
                base=base_states,
                logits=adapted,
                labels=target_labels,
                seed=config.seed,
                mask=mask,
            )
            total = total + config.stab.weight * l_stab
            terms["stab"] = l_stab
            terms.update(m)

    return total, {"total": total, **terms}


__all__ = [
    "A2Config",
    "ADAPTER_PREFIX",
    "CPCConfig",
    "HEPConfig",
    "POSITIVES",
    "StabilityConfig",
    "a2_objective",
    "adapter_params",
    "cpc_term",
    "init_a2_params",
    "init_cpc_params",
    "init_hep_params",
    "param_counts",
    "stability_term",
    "token_mask",
]