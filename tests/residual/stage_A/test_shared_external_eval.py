"""Shared per-checkpoint external evaluation in stages/stage_base.py (used by A1, A2, A3)."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import stages.stage_base as sb
from metrics.performance import cross_entropy_loss, kl_to_base
from stages.stage_base import (
    Domain,
    attach_task_probes,
    evaluate,
    evaluate_task,
    make_eval_fn,
    probe_examples_from_split,
    summarize,
)

V, D, L = 20, 8, 6


class _Fwd:
    def __init__(self, logits):
        self.logits = logits


class ToySubstrate:
    architecture = SimpleNamespace(
        hidden_size=D, num_layers=L, max_position_embeddings=64
    )

    def __init__(self):
        k1, k2 = jax.random.split(jax.random.PRNGKey(7))
        self.emb = jax.random.normal(k1, (V, D)) * 0.5
        self.w = jax.random.normal(k2, (L, D, D)) * 0.3

    def run_with_interception(self, ids, modify_fn=None, intercept_layers=None):
        h = self.emb[ids]
        for layer in range(L):
            h = h + jnp.tanh(h @ self.w[layer])
            if intercept_layers and layer in intercept_layers and modify_fn is not None:
                h = modify_fn(h, layer)
        return _Fwd(h @ self.emb.T)

    def compute_loss(self, logits, ids, losses=("CE",), base_logits=None):
        if tuple(losses) == ("KL",):
            return kl_to_base(base_logits, logits)
        return cross_entropy_loss(logits, ids)


def _probe_logits(preds, vocab=6):
    def fn(params, ids):
        out = jnp.zeros((ids.shape[0], ids.shape[1], vocab))
        for i, p in enumerate(preds):
            out = out.at[i, 0, p].set(5.0)
        return out

    return fn


def _examples(targets, seq=4):
    return [(jnp.zeros((seq,), jnp.int32), t, 0) for t in targets]


def test_evaluate_task_single_label() -> None:
    out = evaluate_task(
        None,
        None,
        None,
        _examples([0, 1, 2, 1]),
        [[0], [1], [2]],
        logits_fn=_probe_logits([0, 1, 2, 0]),
    )
    assert out["task_accuracy"] == pytest.approx(0.75)
    assert out["task_micro_f1"] == pytest.approx(0.75)
    assert 0.0 <= out["task_f1"] <= 1.0
    assert out["task_ece"] == pytest.approx(
        abs(out["task_mean_confidence"] - 0.75), abs=1e-6
    )


def test_evaluate_task_multilabel_and_empty() -> None:
    out = evaluate_task(
        None,
        None,
        None,
        _examples([(0, 1), (2,), (1,)]),
        [[0], [1], [2]],
        logits_fn=_probe_logits([1, 0, 1]),
    )
    assert out["task_accuracy"] == pytest.approx(2 / 3)
    assert out["task_micro_precision"] == pytest.approx(2 / 3)
    assert out["task_micro_recall"] == pytest.approx(2 / 4)
    assert evaluate_task(None, None, None, [], [[0]], logits_fn=_probe_logits([])) == {}


def test_evaluate_reports_token_accuracy_and_pooled_ece() -> None:
    from residual import ResidualAdapter, ResidualConfig

    sub = ToySubstrate()
    adapter = ResidualAdapter(ResidualConfig(rank=2, layers=[3, 4]), sub.architecture)
    params = adapter.init_params()
    rng = np.random.default_rng(0)
    batches = [jnp.asarray(rng.integers(0, V, (2, 8)), jnp.int32) for _ in range(3)]
    out = evaluate(sub, adapter, params, batches, jit=False)
    assert 0.0 <= out["token_accuracy"] <= 1.0 and 0.0 <= out["ece"] <= 1.0
    assert out["kl"] == pytest.approx(0.0, abs=1e-6)
    empty = evaluate(sub, adapter, params, [], jit=False)
    assert empty["ece"] == 0.0 and empty["token_accuracy"] == 0.0


def _row(loss, **extra):
    return {"loss": loss, "ppl": float(np.exp(loss)), "kl": 0.1, **extra}


def test_summarize_adds_matrices_transfer_and_bwt() -> None:
    names = ["a", "b", "c"]
    init = {n: {"base_loss": 3.0, **_row(3.0)} for n in names}
    ex = {"ece": 0.1, "token_accuracy": 0.4}
    rows = [
        {"a": _row(2.0, **ex), "b": _row(2.8, **ex), "c": _row(2.9, **ex)},
        {"a": _row(2.2, **ex), "b": _row(2.0, **ex), "c": _row(2.5, **ex)},
        {"a": _row(2.3, **ex), "b": _row(2.1, **ex), "c": _row(2.0, **ex)},
    ]
    out = summarize(names, init, rows)
    assert out["token_accuracy"][0][0] == 0.4 and out["ece"][2][0] == 0.1
    assert out["backward_transfer_loss"][0] is None
    assert out["backward_transfer_loss"][2] == pytest.approx(-(0.3 + 0.1) / 2)
    assert out["backward_transfer_accuracy"] is None and out["task_micro_f1"] is None
    assert out["heldout_transfer"]["final"]["test_on"] == "c"
    assert out["heldout_transfer"]["final"]["gain"] == pytest.approx(0.5)
    # existing keys are preserved
    assert {
        "mean_forgetting",
        "accuracy_forgetting",
        "task_accuracy",
        "improvement",
    } <= set(out)


def test_summarize_single_joint_task_is_well_defined() -> None:
    out = summarize(
        ["joint"], {"joint": {"base_loss": 3.0, **_row(3.0)}}, [{"joint": _row(2.0)}]
    )
    assert out["mean_forgetting"] == [None]
    assert out["backward_transfer_loss"] is None and out["heldout_transfer"] is None


def test_load_task_examples_multilabel(monkeypatch) -> None:
    rows = [
        {"text": "alpha", "labels": [3, 1]},
        {"text": "beta", "labels": [1]},
        {"text": "gamma", "labels": []},
    ]

    class FakeDS(list):
        column_names = ["text", "labels"]

        def __getitem__(self, k):
            return (
                [r[k] for r in self]
                if isinstance(k, str)
                else list.__getitem__(self, k)
            )

    class Tok:
        pad_token_id = 0
        eos_token_id = 0

        def encode(
            self, text, add_special_tokens=False, truncation=False, max_length=None
        ):
            return [len(text) % 5 + 1]

    monkeypatch.setattr(sb, "load_dataset", lambda *a, **k: FakeDS(rows))
    cfg = sb.DomainConfig(name="eurlex", dataset="x", label_field="labels")
    examples, label_tokens = sb.load_task_examples(
        SimpleNamespace(tokenizer=Tok()), cfg, "test", 4
    )
    assert len(label_tokens) == 2
    assert [e[1] for e in examples] == [(0, 1), (0,)]


def _split(multilabel=False, n=3):
    mask = np.zeros((n, 8), bool)
    mask[:, :5] = True
    return SimpleNamespace(
        input_ids=np.ones((n, 8), np.int32),
        attention_mask=mask,
        targets=np.eye(n, 3, dtype=np.int8),
        label_names=("x", "y", "z"),
        multilabel=multilabel,
        probe_targets=lambda: np.arange(n) % 3,
    )


TOK = SimpleNamespace(encode=lambda text, add_special_tokens=False: [len(text)])


def test_probe_examples_from_split_single_and_multilabel() -> None:
    ex, tokens = probe_examples_from_split(_split(), TOK)
    assert [e[1] for e in ex] == [0, 1, 2] and all(e[2] == 4 for e in ex)
    assert tokens == ((2,), (2,), (2,))
    ex2, _ = probe_examples_from_split(_split(multilabel=True), TOK)
    assert [e[1] for e in ex2] == [(0,), (1,), (2,)]


def _domain(name):
    return Domain(
        name=name,
        train_blocks=np.zeros((2, 4), np.int32),
        eval_set=(jnp.zeros((2, 4), jnp.int32),),
        epochs=1,
    )


def test_attach_task_probes_uses_validation_split_and_reports_skips(
    monkeypatch, tmp_path
) -> None:
    import runtime.data_loader as dl

    monkeypatch.setattr(dl, "load_runtime_config", lambda path: {})
    monkeypatch.setattr(
        dl,
        "legal_task_specs",
        lambda cfg: [SimpleNamespace(name="ok"), SimpleNamespace(name="bad")],
    )
    seen = []

    def fake_load(spec, split, tokenizer, *, sequence_length, max_examples):
        seen.append((spec.name, split, max_examples))
        if spec.name == "bad":
            raise RuntimeError("dataset unreachable")
        return _split()

    monkeypatch.setattr(dl, "load_labeled_split", fake_load)
    cfg = tmp_path / "e.yaml"
    cfg.write_text("probe:\n  stage_max_examples: 7\n", encoding="utf-8")
    monkeypatch.setenv("BENCHMARK_EVAL_CONFIG", str(cfg))
    data = SimpleNamespace(tasks=(_domain("ok"), _domain("bad"), _domain("stories")))
    data = SimpleNamespace(
        **{**data.__dict__}
    )  # replace() needs a dataclass; emulate below
    from dataclasses import dataclass

    @dataclass(frozen=True)
    class RD:
        tasks: tuple

    out = attach_task_probes(RD(tasks=data.tasks), TOK, "x.yaml", seq_len=8)
    by = {t.name: t for t in out.tasks}
    assert seen == [("ok", "validation", 7), ("bad", "validation", 7)]
    assert len(by["ok"].task_examples) == 3 and by["ok"].task_skip_reason is None
    assert by["bad"].task_examples == () and "unreachable" in by["bad"].task_skip_reason
    assert by["stories"].task_examples == () and by["stories"].task_skip_reason is None

    monkeypatch.setenv("BENCHMARK_EVAL_CONFIG", str(tmp_path / "none.yaml"))
    cfg.write_text("probe:\n  enabled: false\n", encoding="utf-8")
    monkeypatch.setenv("BENCHMARK_EVAL_CONFIG", str(cfg))
    rd = RD(tasks=data.tasks)
    assert attach_task_probes(rd, TOK, "x.yaml", seq_len=8) is rd


def test_make_eval_fn_adds_probe_metrics_per_task_and_honours_hooks() -> None:
    task = replace(
        _domain("ok"),
        task_examples=tuple(_examples([0, 1, 2])),
        label_token_ids=((0,), (1,), (2,)),
    )
    skipped = replace(_domain("bad"), task_skip_reason="no labels")
    plain = _domain("stories")
    calls = []

    def lm(params, eval_set):
        calls.append(("lm", params))
        return {"loss": 1.0}

    def logits_fn(params, ids):
        calls.append(("logits", params))
        return _probe_logits([0, 1, 2])(params, ids)

    fn = make_eval_fn(
        None, None, [task, skipped, plain], evaluate_fn=lm, logits_fn=logits_fn
    )
    ok = fn("P", task.eval_set)
    assert ok["loss"] == 1.0 and ok["task_accuracy"] == pytest.approx(1.0)
    assert ("lm", "P") in calls and ("logits", "P") in calls
    assert fn("P", skipped.eval_set)["task_skip_reason"] == "no labels"
    assert set(fn("P", plain.eval_set)) == {"loss"}
    assert set(fn("P", (jnp.zeros((1, 1)),))) == {
        "loss"
    }  # unknown eval set: LM metrics only
