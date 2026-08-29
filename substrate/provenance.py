"""Checkpoint provenance: pinning a substrate to a specific, reconstructible
HuggingFace revision rather than a mutable model_id string.

A ``model_id`` such as ``"gpt2"`` names a repository, not a fixed set of
weights: the tip of that repository can change over time, and a bare
``from_pretrained(model_id)`` call re-resolves whatever "main" currently
means on every call. This module resolves a requested revision to its
concrete commit SHA once, at load time, so the resulting
``FrozenJAXSubstrate`` can record -- and later, callers can verify -- exactly
which checkpoint it was built from.

Mirrors the ``MemoryStatus`` pattern in :mod:`substrate.memory`: resolution
failures (no network, private repo, unknown revision) never raise here. They
are reported as an unresolved, diagnosable status instead, because a
provenance record that might be silently wrong is worse than one that
honestly says it could not be confirmed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CheckpointProvenance:
    """Identity of the checkpoint a substrate was built from.

    ``resolved_sha`` is the pinned commit hash actually used to load both
    the config and the weights (see :func:`resolve_checkpoint_provenance`
    and ``loader.load_substrate_from_hf``). When ``resolved`` is False, no
    commit could be confirmed and ``resolved_sha`` is None -- the substrate
    may still be usable (e.g. offline, cache-only environments), but its
    provenance is not verifiable and this is surfaced rather than hidden.
    """

    model_id: str
    requested_revision: str
    resolved_sha: str | None
    resolved: bool
    diagnostic: str

    def as_dict(self) -> dict[str, str | bool | None]:
        return {
            "model_id": self.model_id,
            "requested_revision": self.requested_revision,
            "resolved_sha": self.resolved_sha,
            "resolved": self.resolved,
            "diagnostic": self.diagnostic,
        }


def resolve_checkpoint_provenance(
    model_id: str, revision: str | None = None
) -> CheckpointProvenance:
    """Resolve ``model_id``[@``revision``] to a pinned commit SHA.

    ``revision`` may be a branch name, tag, or commit SHA; it defaults to
    ``"main"``. Never raises: any failure to resolve (missing
    ``huggingface_hub``, no network, unknown revision, private repo without
    credentials) is reported via ``resolved=False`` and a diagnostic
    message, not an exception, so callers can decide for themselves whether
    an unverifiable checkpoint is acceptable for their use case.
    """
    requested = revision or "main"

    try:
        from huggingface_hub import model_info
    except ImportError as exc:
        return CheckpointProvenance(
            model_id=model_id,
            requested_revision=requested,
            resolved_sha=None,
            resolved=False,
            diagnostic=f"huggingface_hub is not importable: {exc}",
        )

    try:
        info = model_info(model_id, revision=requested)
    except Exception as exc:  # network error, unknown revision, private repo, ...
        return CheckpointProvenance(
            model_id=model_id,
            requested_revision=requested,
            resolved_sha=None,
            resolved=False,
            diagnostic=(
                f"could not resolve {model_id!r}@{requested!r} to a commit: {exc}"
            ),
        )

    sha = getattr(info, "sha", None)
    if not sha:
        return CheckpointProvenance(
            model_id=model_id,
            requested_revision=requested,
            resolved_sha=None,
            resolved=False,
            diagnostic=(
                f"model_info() returned no commit sha for "
                f"{model_id!r}@{requested!r}"
            ),
        )

    return CheckpointProvenance(
        model_id=model_id,
        requested_revision=requested,
        resolved_sha=sha,
        resolved=True,
        diagnostic="ok",
    )