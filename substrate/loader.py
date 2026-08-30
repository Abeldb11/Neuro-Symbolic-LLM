"""Loading of HuggingFace checkpoints into JAX parameter PyTrees.

The JAX parameter trees produced here keep the exact HuggingFace dotted
parameter names (nested), so weights convert without renaming.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp

from .architecture import Architecture, detect_architecture
from .conversion import ConversionReport, convert_state_dict
from .provenance import CheckpointProvenance, resolve_checkpoint_provenance
from .substrate import FrozenJAXSubstrate

_SUPPORTED_FAMILIES = {"gpt2", "gpt_neox"}


def _nested_from_dotted(mapping: Mapping[str, Any]) -> Any:


    def _insert(root: dict[str, Any], parts: list[str], value: Any) -> None:
        node = root
        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                node[part] = value
                return
            if part.isdigit():
                node.setdefault(part, {})
                node = node[part]
            else:
                next_node = node.setdefault(part, {})
                node = next_node

    root: dict[str, Any] = {}
    for dotted_key, value in mapping.items():
        _insert(root, dotted_key.split("."), value)

    def _to_list(node: Any) -> Any:
        if not isinstance(node, dict):
            return node
        if node and all(k.isdigit() for k in node):
            return [_to_list(node[k]) for k in sorted(node, key=int)]
        return {k: _to_list(v) for k, v in node.items()}

    return _to_list(root)


def state_dict_to_jax_pytree(
    state_dict: Mapping[str, Any],
    dtype: Any = jnp.float32,
    dedupe: bool = True,
    max_workers: int | None = None,
    return_report: bool = False,
) -> Any:
    """Convert a torch/HF state dict (or plain numpy mapping) into a JAX
    parameter PyTree matching the Flax model conventions.

    All floating-point leaves are cast to ``dtype`` (float32 by default):
    some checkpoints (e.g. Pythia) ship float16 weights, and computing
    LayerNorm statistics or attention in half precision diverges from the
    reference fp32 forward pass. Pass ``dtype=None`` to preserve each
    tensor's original dtype instead (zero-copy DLPack path; roughly halves
    memory for fp16-native checkpoints) -- see
    :mod:`substrate.conversion` for the full tradeoff this makes; only use
    it once the numerical behavior for the target architecture family is
    understood.

    Tied tensors (e.g. GPT-2's shared input/output embedding) are
    deduplicated to a single JAX buffer by default (``dedupe=True``),
    matching the checkpoint's real memory layout. Non-floating-point
    entries (persistent buffers some checkpoints still carry) are detected
    and excluded rather than silently force-cast into a trainable-looking
    parameter.

    ``return_report=False`` (default) returns exactly the PyTree, matching
    every existing call site's expectation. Pass ``return_report=True`` to
    additionally get a :class:`~substrate.conversion.ConversionReport`
    back as ``(pytree, report)``, for auditing what actually happened
    (dedup savings, excluded keys, timing) on a real checkpoint.
    """
    converted, report = convert_state_dict(
        state_dict, dtype=dtype, dedupe=dedupe, max_workers=max_workers
    )
    pytree = _nested_from_dotted(converted)
    return (pytree, report) if return_report else pytree


def _hf_model_class(model_family: str):
    if model_family == "gpt2":
        from transformers import GPT2LMHeadModel  # local import: heavy dep

        return GPT2LMHeadModel
    if model_family == "gpt_neox":
        from transformers import GPTNeoXForCausalLM  # local import: heavy dep

        return GPTNeoXForCausalLM
    raise ValueError(f"Unsupported model family: {model_family}")


def load_substrate_from_hf(
    model_id: str,
    revision: str | None = None,
    intercept_layers: list[int] | None = None,
    modify_hook: Callable[[jax.Array, int], jax.Array] | None = None,
) -> FrozenJAXSubstrate:
   
    from transformers import AutoConfig, AutoModelForCausalLM  # local import

    provenance = resolve_checkpoint_provenance(model_id, revision)
    pinned_revision = provenance.resolved_sha or provenance.requested_revision

    config = AutoConfig.from_pretrained(model_id, revision=pinned_revision)
    if config.model_type not in _SUPPORTED_FAMILIES:
        raise ValueError(
            f"Unsupported model architecture {config.model_type!r} for model "
            f"{model_id!r}. Supported: {sorted(_SUPPORTED_FAMILIES)}."
        )

    torch_model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=pinned_revision
    )
    torch_model.eval()
    state_dict = torch_model.state_dict()
    params = state_dict_to_jax_pytree(state_dict)
    return FrozenJAXSubstrate(
        params=params,
        config=config,
        intercept_layers=intercept_layers,
        modify_hook=modify_hook,
        provenance=provenance,
    )


def build_substrate_from_state_dict(
    state_dict: Mapping[str, Any],
    config: Any = None,
    intercept_layers: list[int] | None = None,
    modify_hook: Callable[[jax.Array, int], jax.Array] | None = None,
    provenance: CheckpointProvenance | None = None,
) -> FrozenJAXSubstrate:
    """Build a substrate directly from a state dict (HF naming) and an
    optional HF config object.

    This entry point never talks to the Hub, so it cannot resolve
    provenance itself -- pass an already-resolved ``CheckpointProvenance``
    (e.g. one you resolved earlier, or attached when snapshotting a state
    dict to disk) if this checkpoint's identity needs to stay verifiable.
    """
    params = state_dict_to_jax_pytree(state_dict)
    return FrozenJAXSubstrate(
        params=params,
        config=config,
        intercept_layers=intercept_layers,
        modify_hook=modify_hook,
        provenance=provenance,
    )


def architecture_from_params(params: Any, config: Any = None) -> Architecture:
    """Convenience alias for architecture detection."""
    return detect_architecture(params, config)