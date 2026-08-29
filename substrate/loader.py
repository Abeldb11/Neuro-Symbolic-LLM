"""Loading of HuggingFace checkpoints into JAX parameter PyTrees.

The JAX parameter trees produced here keep the exact HuggingFace dotted
parameter names (nested), so weights convert without renaming.
"""


from __future__ import annotations

from .architecture import Architecture, detect_architecture
from .provenance import CheckpointProvenance, resolve_checkpoint_provenance 
from .substrate import FrozenJAXSubstrate

from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp

from .architecture import Architecture, detect_architecture
from .substrate import FrozenJAXSubstrate

_SUPPORTED_FAMILIES = {"gpt2", "gpt_neox"}


def _nested_from_dotted(mapping: Mapping[str, Any]) -> Any:
    """Convert ``{"a.b.c": value}`` into ``{"a": {"b": {"c": value}}}``.

    ``layers.{i}`` segments become list entries so the layer count can be
    discovered from the sequence length.
    """

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


def state_dict_to_jax_pytree(state_dict: Mapping[str, Any]) -> Any:
    """Convert a torch/HF state dict (or plain numpy mapping) into a JAX
    parameter PyTree matching the Flax model conventions.

    All leaves are cast to float32: some checkpoints (e.g. Pythia) ship
    float16 weights, and computing LayerNorm statistics or attention in
    half precision diverges from the reference fp32 forward pass.
    """

    def _to_jax(value: Any) -> jax.Array:
        if isinstance(value, jax.Array):
            return value.astype(jnp.float32)
        arr = value.detach().cpu().numpy() if hasattr(value, "detach") else value
        return jnp.asarray(arr, dtype=jnp.float32)

    jax_dict = {k: _to_jax(v) for k, v in state_dict.items()}
    return _nested_from_dotted(jax_dict)


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

    from transformers import AutoConfig, AutoModelForCausalLM  

    provenance = resolve_checkpoint_provenance(model_id, revision)
    # Fall back to the raw requested revision (e.g. "main") when resolution
    # failed, so loading can still proceed offline/cache-only -- the
    # substrate's `.provenance` honestly records that the pin is unverified,
    # it does not pretend a resolution happened.
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
