"""External-metric layer in benchmark/core.py (toy model; no datasets or GPU)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import benchmark.core as core

V, T = 11, 12


def _deployed(params, ids):
    """Toy deployed model: base logits are fixed; 'params' is a scalar shift toward the label."""
    base = jax.nn.one_hot(ids % V, V) * 2.0
    adapted = base + float(params) * jax.nn.one_hot((ids + 1) % V, V)
    return base, adapted


def _ce(logits, ids):
    logp = jax.nn.log_softmax(logits[:, :-1], axis=-1)
    tgt = ids[:, 1:]
    return float(-jnp.mean(jnp.take_along_axis(logp, tgt[..., None], axis=-1)))


def _batches(seed, n=2, seq=T):
    rng = np.random.default_rng(seed)
    # next token = current + 1, so a positive shift helps
    start = rng.integers(0, V, (2, 1))
    ids = (start + np.arange(seq)) % V
    return tuple(jnp.asarray(ids, jnp.int32) for _ in range(n))


def _run(tasks, params, domains, **kw):
    sets = {d: _batches(i) for i, d in enumerate(domains)}
    args = dict(
        ce=_ce, deployed=_deployed, checkpoint_tasks=tasks, checkpoint_params=params,
        test_sets=sets, probe_sets={}, probe_status={}, anchor_batches=None,
        anchor_reason="off", long_range_sets={}, long_range_reason="off",
    )
    args.update(kw)
    return core._external_evaluation(**args)


def test_sequential_matrices_and_transfer() -> None:
    out = _run(["a", "b", "c"], [0.5, 1.0, 1.5], ["a", "b", "c"])
    assert out["domains"] == ["a", "b", "c"] and out["unseen_domains"] == []
    for key in ("loss", "token_accuracy", "ece", "kl"):
        m = np.asarray(out["matrices"][key])
        assert m.shape == (3, 3) and np.isfinite(m).all()
    assert out["matrices"]["task_accuracy"] is None  # no probe configured
    # Larger shift helps this toy task, so later checkpoints reduce loss: BWT (loss) > 0.
    assert out["backward_transfer_loss"][0] is None and out["backward_transfer_loss"][2] > 0
    assert out["backward_transfer_accuracy"] is None
    assert out["heldout_transfer"]["final"]["test_on"] == "c"
    assert out["unseen_transfer"]["status"] == "unavailable"
    assert out["anchor"]["status"] == "unavailable" and out["long_range_proxy"]["status"] == "unavailable"
    json.dumps(out, default=float)


def test_heldout_protocol_has_unseen_transfer_and_non_square_skips_bwt() -> None:
    out = _run(["a", "b"], [0.5, 1.0], ["a", "b", "z"])
    assert out["trained_order"] == ["a", "b"] and out["unseen_domains"] == ["z"]
    assert out["unseen_transfer"]["status"] == "measured"
    assert out["unseen_transfer"]["mean_gain"] > 0  # shift generalises in the toy model
    assert out["backward_transfer_loss"][1] is not None  # 2 trained x 2 checkpoints is square


def test_joint_run_has_one_checkpoint_and_no_continual_metrics() -> None:
    out = _run(["joint"], [1.0], ["a", "b"])
    assert out["trained_order"] == [] and np.asarray(out["matrices"]["loss"]).shape == (1, 2)
    assert out["backward_transfer_loss"] is None
    assert out["heldout_transfer"]["status"] == "unavailable"
    assert out["unseen_transfer"]["status"] == "unavailable"


def test_anchor_and_long_range_proxy_measured() -> None:
    out = _run(
        ["a", "b"], [0.5, 1.0], ["a", "b"],
        anchor_batches=_batches(9), long_range_sets={"a": _batches(5, seq=24)},
    )
    assert out["anchor"]["status"] == "measured" and len(out["anchor"]["kl"]) == 2
    assert out["anchor"]["final_kl"] >= 0
    lr = out["long_range_proxy"]
    assert lr["status"] == "measured" and "delta_vs_base" in lr["per_domain"]["a"]


def test_label_token_probe_single_and_multilabel() -> None:
    rng = np.random.default_rng(0)
    ids = rng.integers(0, V, (6, T)).astype(np.int32)
    positions = np.full(6, T - 1, np.int32)
    token_matrix = np.array([[1], [2], [3]], np.int32)
    # adapted logits at the last position favour (ids+1)%V; the label tokens are 1,2,3
    probe = {"ids": ids, "positions": positions, "token_matrix": token_matrix,
             "targets": [0, 1, 2, 0, 1, 2], "multilabel": False}
    res = core._run_external_probe(_deployed, 1.0, probe, batch_size=4)
    assert set(res) >= {"task_accuracy", "task_f1", "task_micro_f1", "task_ece"}
    assert res["task_micro_f1"] == res["task_accuracy"]
    assert all(0.0 <= v <= 1.0 for v in res.values())
    multi = dict(probe, targets=[(0, 1), (1,), (2,), (0,), (1, 2), (2,)], multilabel=True)
    res2 = core._run_external_probe(_deployed, 1.0, multi, batch_size=4)
    assert 0.0 <= res2["task_accuracy"] <= 1.0 and "task_ece" in res2


def test_probe_sets_use_pinned_specs_and_report_skips(monkeypatch) -> None:
    import runtime.data_loader as dl

    specs = [SimpleNamespace(name="ok"), SimpleNamespace(name="bad")]
    monkeypatch.setattr(dl, "legal_task_specs", lambda cfg: specs)

    def fake_load(spec, split, tokenizer, *, sequence_length, max_examples):
        if spec.name == "bad":
            raise RuntimeError("dataset unreachable")
        mask = np.zeros((3, sequence_length), bool)
        mask[:, :5] = True
        return SimpleNamespace(
            input_ids=np.ones((3, sequence_length), np.int32), attention_mask=mask,
            targets=np.eye(3, dtype=np.int8), label_names=("x", "y", "z"), multilabel=False,
            probe_targets=lambda: np.array([0, 1, 2]),
        )

    monkeypatch.setattr(dl, "load_labeled_split", fake_load)
    tok = SimpleNamespace(encode=lambda text, add_special_tokens=False: [len(text)])
    sets, status = core._external_probe_sets(tok, {}, {"probe": {"max_examples": 2}}, 8)
    assert list(sets) == ["ok"] and sets["ok"]["positions"].tolist() == [4, 4, 4]
    assert sets["ok"]["token_matrix"].shape == (3, 1)
    assert status["ok"]["status"] == "measured" and status["bad"]["status"] == "skipped"
    assert core._external_probe_sets(tok, {}, {"probe": {"enabled": False}}, 8) == ({}, {})


def test_external_block_never_raises(tmp_path, monkeypatch) -> None:
    out = core._external_block(
        substrate=None, deployed_with_params=None, checkpoint_rows=[], test_sets={},
        training=None, runtime_config_path=tmp_path / "missing.yaml",
    )
    assert out["status"] == "failed" and out["reason"]


def test_settings_loader_override(tmp_path, monkeypatch) -> None:
    cfg = tmp_path / "e.yaml"
    cfg.write_text("probe:\n  max_examples: 7\n", encoding="utf-8")
    monkeypatch.setenv("BENCHMARK_EVAL_CONFIG", str(cfg))
    assert core._load_external_settings()["probe"]["max_examples"] == 7
    monkeypatch.setenv("BENCHMARK_EVAL_CONFIG", str(tmp_path / "none.yaml"))
    assert core._load_external_settings() == {}


def _rec(protocol, loss, acc=None, **kw):
    final = {"loss": loss, **({"task_accuracy": acc} if acc else {})}
    return {
        "run_id": f"r-{protocol}", "method": "a1", "seed": 0, "rank": 4, "k": None,
        "protocol": protocol, "task_order": ["x", "y"], "steps": 10, "tokens": 100,
        "deployed_evaluation": {"external": {"domains": ["x", "y"], "base_loss": [3.0, 3.0], "final": final}},
        **kw,
    }


def test_joint_gap_sign_pairing_and_budget_guard() -> None:
    seq = _rec("sequential", {"x": 2.6, "y": 2.0}, {"x": 0.5, "y": 0.7})
    jnt = _rec("joint", {"x": 2.4, "y": 2.1}, {"x": 0.6, "y": 0.65})
    core._attach_external_joint_gap([seq, jnt])
    gap = seq["external_joint_gap"]
    assert gap["status"] == "measured"
    assert gap["loss"]["gap"] == pytest.approx([0.2, -0.1])  # positive = sequential worse
    assert gap["task_accuracy"]["gap"] == pytest.approx([0.1, -0.05])
    assert "external_joint_gap" not in jnt

    seq2, jnt2 = _rec("sequential", {"x": 1.0, "y": 1.0}), _rec("joint", {"x": 1.0, "y": 1.0}, steps=11)
    core._attach_external_joint_gap([seq2, jnt2])
    assert seq2["external_joint_gap"]["status"] == "invalid_budget_mismatch"

    alone = _rec("sequential", {"x": 1.0, "y": 1.0})
    core._attach_external_joint_gap([alone])
    assert "external_joint_gap" not in alone


def test_report_values_for_external_rows() -> None:
    rec = _rec("sequential", {"x": 2.0, "y": 2.2})
    ext = rec["deployed_evaluation"]["external"]
    ext["final"].update({"token_accuracy": {"x": 0.4, "y": 0.5}, "task_accuracy": {"x": 0.6}})
    ext.update(
        backward_transfer_loss=[None, -0.1, -0.3],
        heldout_transfer={"final": {"gain": 0.3, "ratio": 0.5}},
        unseen_transfer={"mean_gain": 0.2, "mean_ratio": None},
        anchor={"final_kl": 0.2, "peak_kl": 0.4, "final_loss_change": 0.05},
        long_range_proxy={"mean_delta_vs_base": -0.02},
    )
    rec["external_joint_gap"] = {"loss": {"mean_relative_gap": 0.1}}
    get = lambda m: core._report_metric_value(rec, m, 4)  # noqa: E731
    assert get("ext_token_accuracy") == pytest.approx([0.4, 0.5])
    assert get("ext_task_accuracy") == [0.6]
    assert get("ext_bwt_loss") == [-0.3]
    assert get("ext_zero_shot_gain") == [0.3] and get("ext_zero_shot_ratio") == [0.5]
    assert get("ext_unseen_gain") == [0.2] and get("ext_unseen_ratio") == []
    assert get("ext_anchor_kl_final") == [0.2] and get("ext_anchor_kl_peak") == [0.4]
    assert get("ext_joint_gap_relative") == [0.1] and get("ext_long_range_proxy") == [-0.02]
    assert get("ext_task_f1") == []  # not measured -> N/A, never zero


def test_performance_additions() -> None:
    from metrics.performance import multilabel_hit_metrics, unseen_domain_transfer

    m = multilabel_hit_metrics([{0, 1}, {2}], [0, 1], 3)
    assert m["hit_at_1"] == pytest.approx(0.5)
    out = unseen_domain_transfer([2.0, 2.5, 2.9], [3.0, 3.0, 3.0], [0, 1], [2])
    assert out["gain"][2] == pytest.approx(0.1)
    assert out["ratio"][2] == pytest.approx(0.1 / 0.75)
    assert unseen_domain_transfer([3.0, 3.0, 2.9], [3.0, 3.0, 3.0], [0, 1], [2])["ratio"][2] is None