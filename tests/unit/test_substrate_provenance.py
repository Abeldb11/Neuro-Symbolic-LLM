"""Provenance never requires the network in unit tests: huggingface_hub's
model_info is monkeypatched, mirroring this suite's no-live-services policy
for everything under tests/unit."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from substrate import CheckpointProvenance, FrozenJAXSubstrate
from substrate.provenance import resolve_checkpoint_provenance

from ..conftest import GPT2_CFG, _torch_model


def _fake_model_info_ok(repo_id, revision=None):
    return SimpleNamespace(sha="a" * 40)


def _fake_model_info_no_sha(repo_id, revision=None):
    return SimpleNamespace(sha=None)


def _fake_model_info_raises(repo_id, revision=None):
    raise OSError("simulated network failure")


class TestResolveCheckpointProvenance:
    def test_resolves_to_pinned_sha(self, monkeypatch):
        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "model_info", _fake_model_info_ok)
        prov = resolve_checkpoint_provenance("gpt2", revision="main")
        assert prov.resolved is True
        assert prov.resolved_sha == "a" * 40
        assert prov.model_id == "gpt2"
        assert prov.requested_revision == "main"
        assert prov.diagnostic == "ok"

    def test_defaults_revision_to_main(self, monkeypatch):
        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "model_info", _fake_model_info_ok)
        prov = resolve_checkpoint_provenance("gpt2")
        assert prov.requested_revision == "main"

    def test_network_failure_does_not_raise(self, monkeypatch):
        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "model_info", _fake_model_info_raises)
        prov = resolve_checkpoint_provenance("gpt2", revision="main")
        assert prov.resolved is False
        assert prov.resolved_sha == None
        assert "simulated network failure" in prov.diagnostic

    def test_missing_sha_is_unresolved_not_a_crash(self, monkeypatch):
        import huggingface_hub

        monkeypatch.setattr(huggingface_hub, "model_info", _fake_model_info_no_sha)
        prov = resolve_checkpoint_provenance("gpt2", revision="main")
        assert prov.resolved is False
        assert prov.resolved_sha is None

    def test_as_dict_roundtrips(self):
        prov = CheckpointProvenance(
            model_id="gpt2",
            requested_revision="main",
            resolved_sha="a" * 40,
            resolved=True,
            diagnostic="ok",
        )
        d = prov.as_dict()
        assert d == {
            "model_id": "gpt2",
            "requested_revision": "main",
            "resolved_sha": "a" * 40,
            "resolved": True,
            "diagnostic": "ok",
        }


class TestSubstrateProvenanceIntegration:
    def _substrate_with_provenance(self, prov: CheckpointProvenance | None):
        from transformers import GPT2Config

        model = _torch_model("gpt2")
        model.eval()
        from substrate import state_dict_to_jax_pytree

        params = state_dict_to_jax_pytree(model.state_dict())
        return FrozenJAXSubstrate(
            params, GPT2Config(**GPT2_CFG), provenance=prov
        )

    def test_provenance_none_by_default(self):
        sub = self._substrate_with_provenance(None)
        assert sub.provenance is None
        assert "unknown" in repr(sub)

    def test_resolved_provenance_surfaces_in_repr(self):
        prov = CheckpointProvenance(
            model_id="gpt2",
            requested_revision="main",
            resolved_sha="a" * 40,
            resolved=True,
            diagnostic="ok",
        )
        sub = self._substrate_with_provenance(prov)
        assert sub.provenance is prov
        assert "gpt2@aaaaaaaaaaaa" in repr(sub)  # first 12 chars of the sha

    def test_unresolved_provenance_is_visible_not_hidden(self):
        prov = CheckpointProvenance(
            model_id="gpt2",
            requested_revision="main",
            resolved_sha=None,
            resolved=False,
            diagnostic="offline",
        )
        sub = self._substrate_with_provenance(prov)
        assert "UNRESOLVED" in repr(sub)

    def test_verify_frozen_includes_provenance(self):
        prov = CheckpointProvenance(
            model_id="gpt2",
            requested_revision="main",
            resolved_sha="b" * 40,
            resolved=True,
            diagnostic="ok",
        )
        sub = self._substrate_with_provenance(prov)
        report = sub.verify_frozen()
        assert report["provenance"] == prov.as_dict()

    def test_verify_frozen_provenance_none_when_unset(self):
        sub = self._substrate_with_provenance(None)
        report = sub.verify_frozen()
        assert report["provenance"] is None
