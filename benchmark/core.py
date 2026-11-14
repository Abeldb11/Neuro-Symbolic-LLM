"""Small, artifact-driven A1/A2/A3 benchmark harness."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from benchmark.provenance import (
    effective_condition,
    finalize_attempt,
    next_attempt_name,
    sha256_file,
    source_fingerprint,
    validate_complete_attempt,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNTIME_CONFIG = ROOT / "configs" / "runtime_datasets.yaml"
BENCHMARK_RUNTIME_SOURCE = "separate"
BENCHMARK_LEARNING_RATE = 1.0e-3
DEFAULT_CONFIGS = {
    "a1": ROOT / "configs" / "stage_A" / "a1_lora_baseline.yaml",
    "a2": ROOT / "configs" / "stage_A" / "a2_pc_inspired.yaml",
    "a3": ROOT / "configs" / "stage_A" / "a3_hybrid_pc.yaml",
}
RUNNERS = {
    "a1": ROOT / "experiments" / "run_stage_a1.py",
    "a2": ROOT / "experiments" / "run_stage_a2.py",
    "a3": ROOT / "experiments" / "run_stage_a3.py",
}
EXPECTED_DOMAINS = ("eurlex", "ledgar", "scotus")
MEASURED_METRICS = (
    "loss_matrix",
    "perplexity",
    "mean_forgetting",
    "optimizer_steps",
    "tokens",
    "mean_step_time",
    "estimated_tokens_per_second",
    "params_unchanged",
    "residual_compressibility",
    "effective_rank",
    "invariant_redundancy",
    "task_gradient_conflict",
    "token_accuracy",
    "expected_calibration_error",
    "kl_to_base",
    "paired_seed_delta",
    "parameter_update_commutator",
    "operator_update_commutator",
    "update_commutator",
    "heldout_transfer",
    "sequential_joint_gap",
    "anchor_kl",
    "long_range_coherence",
    "inference_latency",
    "peak_memory",
)
A4_SYMBOLIC_METRICS = (
    "retrieval_precision_recall",
    "retrieval_hit_rate",
    "retrieval_mean_reciprocal_rank",
    "retrieval_activation_rate",
    "tier3_trigger_rate",
    "template_coverage",
    "key_alignment_nll",
    "value_alignment_error",
    "query_cluster_silhouette",
    "template_factor_fracture",
    "template_factor_overlap",
    "symbolic_key_entanglement",
    "symbolic_head_capacity",
    "symbolic_alignment_preservation",
    "symbolic_attention_contribution",
    "symbolic_gradient_conflict",
    "symbolic_mixed_fisher_coupling",
    "symbolic_update_commutator",
    "human_template_review",
    "bank_growth_stability",
    "consolidation_queue_metrics",
    "cross_tier_latency",
    "cuda_transfer_latency",
)
LATER_STAGE_SYMBOLIC_METRICS = (
    "bank_template_mutual_information",
    "column_template_binding_stability",
)
EXTERNAL_METRIC_SPECS = (
    ("Validation loss", "validation_loss", "lower"),
    ("Perplexity", "perplexity", "lower"),
    ("Held-out test loss", "test_loss", "lower"),
    ("Held-out test perplexity", "test_perplexity", "lower"),
    ("Deployed LM loss", "deployed_loss", "lower"),
    ("Deployed LM perplexity", "deployed_perplexity", "lower"),
    ("Token accuracy", "token_accuracy", "higher"),
    ("Token calibration ECE", "calibration", "lower"),
    ("On-domain KL drift", "domain_kl", "lower"),
    ("Validation-loss forgetting", "forgetting", "lower"),
    ("Held-out transfer", "heldout_transfer", "higher"),
    ("Sequential versus joint gap", "sequential_joint_gap", "lower"),
    ("Anchor KL drift", "anchor_kl", "lower"),
    ("Long-range coherence accuracy", "long_range", "higher"),
    ("Token accuracy (diagnostic)", "ext_token_accuracy", "higher"),
    ("Task accuracy, label-token probe", "ext_task_accuracy", "higher"),
    ("Task macro-F1, label-token probe", "ext_task_f1", "higher"),
    ("Task micro-F1, label-token probe", "ext_task_micro_f1", "higher"),
    ("Task ECE, label-token probe", "ext_task_ece", "lower"),
    ("Backward transfer, loss (negative = forgetting)", "ext_bwt_loss", "higher"),
    (
        "Backward transfer, accuracy (negative = forgetting)",
        "ext_bwt_accuracy",
        "higher",
    ),
    (
        "Zero-shot gain on the last domain (checkpoint matrix)",
        "ext_zero_shot_gain",
        "higher",
    ),
    ("Zero-shot gain ratio to in-domain gain", "ext_zero_shot_ratio", "higher"),
    ("Gain on never-trained domains (held-out protocol)", "ext_unseen_gain", "higher"),
    (
        "Sequential minus joint, relative gap (guarded)",
        "ext_joint_gap_relative",
        "lower",
    ),
    ("Anchor KL, wikitext, final", "ext_anchor_kl_final", "lower"),
    ("Anchor KL, wikitext, peak", "ext_anchor_kl_peak", "lower"),
    ("Anchor loss change, wikitext, final", "ext_anchor_loss_change", "lower"),
    ("Long-range proxy (context gain vs base)", "ext_long_range_proxy", "higher"),
)
INTERNAL_METRIC_SPECS = (
    ("Residual compressibility", "residual_compressibility", "higher"),
    ("Effective rank", "effective_rank", "higher"),
    ("Invariant redundancy", "invariant_redundancy", "lower"),
    ("Task-gradient conflict", "task_gradient_conflict", "lower"),
    ("Update commutator", "update_commutator", "lower"),
)
OPERATIONAL_METRIC_SPECS = (
    ("Trainable parameters", "trainable_params", "lower"),
    ("Training tokens/second", "tokens_per_second", "higher"),
    ("Inference latency (seconds)", "inference_latency", "lower"),
    ("Peak evaluator memory (bytes)", "peak_memory", "lower"),
)


@dataclass(frozen=True)
class RunSpec:
    method: str
    seed: int
    protocol: str
    run_id: str
    config: str
    output_dir: str
    run_name: str
    k: int | None = None
    rank: int = 16
    runtime_source: str = BENCHMARK_RUNTIME_SOURCE
    runtime_config: str = str(DEFAULT_RUNTIME_CONFIG)
    task_order: tuple[str, ...] = EXPECTED_DOMAINS
    heldout_tasks: tuple[str, ...] = ()
    a2_arm: str | None = None


def _sanitize_value(value: Any) -> Any:
    """Recursively sanitize data structure by converting non-finite floats to None."""
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _sanitize_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_sanitize_value(v) for v in value]
    return value


def _jsonable(value: Any) -> Any:
    return _sanitize_value(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = _jsonable(value)
    path.write_text(
        json.dumps(clean, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _read_json(path: Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


# =============================================================================
# External metrics: per-checkpoint evaluation of the deployed model.
#
# Everything between this banner and ``_load_deployed_metrics`` belongs to the
# external-metric layer (math lives in ``metrics/performance.py``). It is purely
# additive: it reads the same checkpoints/test sets as the rest of the evaluator,
# is wrapped so a failure becomes a status instead of an exception, and writes
# only to ``deployed_evaluation["external"]`` / ``record["external_joint_gap"]``.
# =============================================================================
EXTERNAL_EVAL_SETTINGS = ROOT / "configs" / "benchmark_eval.yaml"


def _load_external_settings() -> dict[str, Any]:
    """Probe / anchor / long-range settings (``BENCHMARK_EVAL_CONFIG`` overrides the path)."""
    path = Path(os.environ.get("BENCHMARK_EVAL_CONFIG", str(EXTERNAL_EVAL_SETTINGS)))
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _external_probe_sets(
    tokenizer: Any,
    runtime_config: dict[str, Any],
    settings: dict[str, Any],
    seq_len: int,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Labelled test examples per legal task, from the pinned ``legal_task`` specs."""
    from runtime.data_loader import legal_task_specs, load_labeled_split

    cfg = settings.get("probe") or {}
    if cfg.get("enabled", True) is False:
        return {}, {}
    max_examples = int(cfg.get("max_examples", 500))
    sets: dict[str, Any] = {}
    status: dict[str, dict[str, Any]] = {}
    for spec in legal_task_specs(runtime_config):
        try:
            split = load_labeled_split(
                spec,
                "test",
                tokenizer,
                sequence_length=seq_len,
                max_examples=max_examples,
            )
            from stages.stage_base import probe_examples_from_split

            examples, token_ids = probe_examples_from_split(split, tokenizer)
            if not examples:
                raise ValueError("no labelled test examples")
            width = max(len(t) for t in token_ids)
            token_matrix = np.asarray(
                [list(t) + [t[-1]] * (width - len(t)) for t in token_ids],
                dtype=np.int32,
            )
            ids: Any = np.stack([np.asarray(e[0], dtype=np.int32) for e in examples])
            positions: Any = np.asarray([e[2] for e in examples], dtype=np.int32)
            targets = [e[1] for e in examples]
            keep = list(range(len(examples)))
            sets[spec.name] = {
                "ids": ids[keep],
                "positions": positions[keep].astype(np.int32),
                "targets": [targets[i] for i in keep],
                "token_matrix": token_matrix,
                "multilabel": bool(split.multilabel),
            }
            status[spec.name] = {
                "status": "measured",
                "n_examples": len(keep),
                "n_classes": len(token_ids),
                "multilabel": bool(split.multilabel),
            }
        except Exception as exc:  # dataset/network problems must not abort the run
            status[spec.name] = {
                "status": "skipped",
                "reason": f"{type(exc).__name__}: {exc}"[:300],
            }
    return sets, status


def _run_external_probe(
    deployed: Any, params: Any, probe: dict[str, Any], batch_size: int = 16
) -> dict[str, float]:
    """Label-token classification probe: accuracy, macro/micro-F1 and class-level ECE."""
    import jax.numpy as jnp

    from metrics.performance import (
        calibration_from_predictions,
        classification_metrics,
        classification_scores,
        multilabel_hit_metrics,
    )

    token_matrix = jnp.asarray(probe["token_matrix"])
    preds, confs = [], []
    for start in range(0, len(probe["ids"]), batch_size):
        ids = jnp.asarray(probe["ids"][start : start + batch_size])
        positions = jnp.asarray(probe["positions"][start : start + batch_size])
        _, logits = deployed(params, ids)
        pred, conf = classification_scores(logits, positions, token_matrix)
        preds.append(np.asarray(pred))
        confs.append(np.asarray(conf))
    predictions, confidence = np.concatenate(preds), np.concatenate(confs)
    targets = probe["targets"]
    if probe["multilabel"]:
        m = multilabel_hit_metrics(targets, predictions, int(token_matrix.shape[0]))
        correct = np.array(
            [int(p in g) for g, p in zip(targets, predictions, strict=True)]
        )
        out = {
            "task_accuracy": m[
                "hit_at_1"
            ],  # hit@1: the predicted label is a gold label
            "task_f1": m["micro_f1"],
            "task_micro_f1": m["micro_f1"],
        }
    else:
        y = np.asarray(targets, dtype=np.int32)
        m = classification_metrics(y, predictions)
        correct = (predictions == y).astype(int)
        # Single-label multiclass: micro-F1 equals accuracy.
        out = {
            "task_accuracy": m["accuracy"],
            "task_f1": m["macro_f1"],
            "task_micro_f1": m["accuracy"],
        }
    ece = calibration_from_predictions(confidence, correct)["ece"]
    if ece is not None:
        out["task_ece"] = float(ece)
    return {k: float(v) for k, v in out.items()}


def _external_anchor_batches(
    substrate: Any, settings: dict[str, Any], training: Any
) -> tuple[Any, str | None]:
    from runtime.data_loader import (
        extract_texts,
        load_dataset_split,
        make_eval_set,
        tokenize_texts,
    )

    cfg = settings.get("anchor")
    if not cfg:
        return None, "no anchor section in benchmark_eval.yaml"
    try:
        dataset = load_dataset_split(
            cfg["dataset"],
            cfg.get("split", "validation"),
            dataset_config=cfg.get("config"),
            revision=cfg.get("revision"),
        )
        texts = extract_texts(dataset, cfg.get("text_field", "text"))
        blocks = tokenize_texts(
            texts,
            substrate.tokenizer,
            sequence_length=training.seq_len,
            separator=cfg.get("separator", "\n\n"),
            task_name="anchor",
        )
        n_batches = max(1, -(-int(cfg.get("max_blocks", 64)) // training.batch_size))
        return make_eval_set(blocks, training.batch_size, n_batches), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"[:300]


def _external_long_range_sets(
    substrate: Any, items: dict[str, dict[str, Any]], settings: dict[str, Any]
) -> tuple[dict[str, Any], str | None]:
    from runtime.data_loader import load_split_texts, make_eval_set, tokenize_texts

    cfg = settings.get("long_range_proxy")
    if not cfg:
        return {}, "no long_range_proxy section in benchmark_eval.yaml"
    seq_len = int(cfg.get("seq_len", 512))
    limit = getattr(substrate.architecture, "max_position_embeddings", None)
    if limit is not None and seq_len > int(limit):
        return {}, f"seq_len {seq_len} exceeds the model context {limit}"
    batch = int(cfg.get("batch_size", 2))
    sets: dict[str, Any] = {}
    try:
        for name, item in items.items():
            texts = load_split_texts(item, item["test_split"])
            blocks = tokenize_texts(
                texts,
                substrate.tokenizer,
                sequence_length=seq_len,
                separator=str(item.get("separator", "\n\n")),
                task_name=name,
                allow_empty=True,
            )
            if len(blocks):
                sets[name] = make_eval_set(
                    blocks, batch, max(1, -(-int(cfg.get("max_blocks", 16)) // batch))
                )
    except Exception as exc:
        return {}, f"{type(exc).__name__}: {exc}"[:300]
    return sets, (
        None if sets else "no domain has enough tokens for the long-range proxy"
    )


def _external_evaluation(
    *,
    ce: Any,
    deployed: Any,
    checkpoint_tasks: list[str],
    checkpoint_params: list[Any],
    test_sets: dict[str, Any],
    probe_sets: dict[str, Any],
    probe_status: dict[str, dict[str, Any]],
    anchor_batches: Any,
    anchor_reason: str | None,
    long_range_sets: dict[str, Any],
    long_range_reason: str | None,
    n_buckets: int = 4,
) -> dict[str, Any]:
    """Every checkpoint on every domain with the deployed model (A3 through its two-pass path).

    ``deployed(params, ids)`` returns ``(base_logits, adapted_logits)``; ``ce(logits, ids)`` the
    cross-entropy. Columns are ordered trained-domains-first (in checkpoint order), then any
    domain never trained on (held-out protocol).
    """
    from metrics.performance import (
        anchor_drift,
        backward_transfer,
        calibration_stats,
        ece_from_stats,
        heldout_transfer,
        kl_to_base,
        long_range_summary,
        position_loss_profile,
        token_accuracy,
        unseen_domain_transfer,
    )

    names = list(test_sets)
    trained: list[str] = []
    for task in checkpoint_tasks:
        if task in test_sets and task not in trained:
            trained.append(task)
    domains = trained + [d for d in names if d not in trained]
    n_ckpt = len(checkpoint_params)

    def domain_row(params: Any, batches: Any) -> dict[str, float]:
        if not batches:
            return {
                k: float("nan")
                for k in ("loss", "token_accuracy", "kl", "ece", "base_loss")
            }
        losses, base_losses, accs, kls = [], [], [], []
        pooled: tuple[Any, Any, Any] | None = None
        for ids in batches:
            base, adapted = deployed(params, ids)
            losses.append(ce(adapted, ids))
            base_losses.append(ce(base, ids))
            accs.append(float(token_accuracy(adapted, ids)))
            kls.append(float(kl_to_base(base, adapted)))
            stats = calibration_stats(adapted, ids)
            pooled = (
                stats
                if pooled is None
                else (pooled[0] + stats[0], pooled[1] + stats[1], pooled[2] + stats[2])
            )
        return {
            "loss": float(np.mean(losses)),
            "base_loss": float(np.mean(base_losses)),
            "token_accuracy": float(np.mean(accs)),
            "kl": float(np.mean(kls)),
            "ece": (
                float(ece_from_stats(*pooled)) if pooled is not None else float("nan")
            ),
        }

    keys = ("loss", "token_accuracy", "ece", "kl")
    task_keys = ("task_accuracy", "task_f1", "task_micro_f1", "task_ece")
    matrices: dict[str, list[list[float]]] = {k: [] for k in keys + task_keys}
    base_loss = [float("nan")] * len(domains)
    for t, params in enumerate(checkpoint_params):
        rows = {d: domain_row(params, test_sets[d]) for d in domains}
        if t == 0:
            base_loss = [rows[d]["base_loss"] for d in domains]
        for k in keys:
            matrices[k].append([rows[d][k] for d in domains])
        probes = {
            d: (
                _run_external_probe(deployed, params, probe_sets[d])
                if d in probe_sets
                else {}
            )
            for d in domains
        }
        for k in task_keys:
            matrices[k].append([float(probes[d].get(k, np.nan)) for d in domains])

    def has_values(m: list[list[float]]) -> bool:
        return bool(np.isfinite(np.asarray(m, dtype=float)).any())

    def or_none(x: float) -> float | None:
        return float(x) if np.isfinite(x) else None

    out: dict[str, Any] = {
        "status": "measured",
        "checkpoints": list(checkpoint_tasks),
        "domains": domains,
        "trained_order": trained,
        "unseen_domains": [d for d in domains if d not in trained],
        "base_loss": base_loss,
        "matrices": {k: (v if has_values(v) else None) for k, v in matrices.items()},
        "probe": probe_status,
        "final": {
            k: dict(zip(domains, v[-1], strict=True))
            for k, v in matrices.items()
            if has_values(v)
        },
    }
    loss_m: Any = np.asarray(matrices["loss"], dtype=float)
    acc_m: Any = np.asarray(matrices["task_accuracy"], dtype=float)
    square = bool(trained) and len(trained) == n_ckpt and n_ckpt >= 2
    reason = "needs one checkpoint per trained domain (sequential or held-out protocol)"
    if square:
        sub_loss, sub_acc = loss_m[:, :n_ckpt], acc_m[:, :n_ckpt]
        sub_base = np.asarray(base_loss[:n_ckpt], dtype=float)
        out["backward_transfer_loss"] = [None] + [
            or_none(backward_transfer(sub_loss, t, higher_is_better=False))
            for t in range(1, n_ckpt)
        ]
        out["backward_transfer_accuracy"] = (
            [None] + [or_none(backward_transfer(sub_acc, t)) for t in range(1, n_ckpt)]
            if np.isfinite(sub_acc).any()
            else None
        )
        out["heldout_transfer"] = heldout_transfer(sub_loss, sub_base, trained)
    else:
        out["backward_transfer_loss"] = out["backward_transfer_accuracy"] = None
        out["heldout_transfer"] = {"status": "unavailable", "reason": reason}
    unseen_idx = [i for i, d in enumerate(domains) if d not in trained]
    if trained and unseen_idx:
        out["unseen_transfer"] = {
            "status": "measured",
            **unseen_domain_transfer(
                loss_m[-1], base_loss, list(range(len(trained))), unseen_idx
            ),
        }
    else:
        out["unseen_transfer"] = {
            "status": "unavailable",
            "reason": "no domain was held out of training",
        }

    if anchor_batches:
        a_kl, a_loss = [], []
        for params in checkpoint_params:
            row = domain_row(params, anchor_batches)
            a_kl.append(row["kl"])
            a_loss.append(row["loss"])
        a_base = domain_row(checkpoint_params[0], anchor_batches)["base_loss"]
        out["anchor"] = {
            "status": "measured",
            "n_batches": len(anchor_batches),
            "base_loss": a_base,
            **anchor_drift(a_kl, a_loss, a_base),
        }
    else:
        out["anchor"] = {
            "status": "unavailable",
            "reason": anchor_reason or "no anchor corpus",
        }

    if long_range_sets:
        final = checkpoint_params[-1]
        per_domain = {}
        for d, batches in long_range_sets.items():
            adapted_profiles, base_profiles = [], []
            for ids in batches:
                base, adapted = deployed(final, ids)
                adapted_profiles.append(position_loss_profile(adapted, ids, n_buckets))
                base_profiles.append(position_loss_profile(base, ids, n_buckets))
            per_domain[d] = long_range_summary(
                np.mean(adapted_profiles, axis=0), np.mean(base_profiles, axis=0)
            )
        out["long_range_proxy"] = {
            "status": "measured",
            "note": "position-bucketed loss proxy; not a long-context benchmark",
            "per_domain": per_domain,
            "mean_delta_vs_base": float(
                np.mean([v["delta_vs_base"] for v in per_domain.values()])
            ),
            "mean_late_loss_change": float(
                np.mean([v["late_loss_change"] for v in per_domain.values()])
            ),
        }
    else:
        out["long_range_proxy"] = {
            "status": "unavailable",
            "reason": long_range_reason or "not configured",
        }
    return out


def _external_block(
    *,
    substrate: Any,
    deployed_with_params: Any,
    checkpoint_rows: list[dict[str, Any]],
    test_sets: dict[str, Any],
    training: Any,
    runtime_config_path: Path,
) -> dict[str, Any]:
    """Entry point used by ``_load_deployed_metrics``; never raises."""
    try:
        from runtime.data_loader import load_runtime_config

        runtime_config = load_runtime_config(runtime_config_path)
        settings = _load_external_settings()
        items = {
            str(d["name"]): d
            for d in runtime_config.get("datasets", [])
            if d.get("type") == "separate" and str(d["name"]) in test_sets
        }
        probe_sets, probe_status = _external_probe_sets(
            substrate.tokenizer, runtime_config, settings, training.seq_len
        )
        anchor_batches, anchor_reason = _external_anchor_batches(
            substrate, settings, training
        )
        long_sets, long_reason = _external_long_range_sets(substrate, items, settings)
        return _external_evaluation(
            ce=lambda logits, ids: float(
                substrate.compute_loss(logits, ids, losses=("CE",))
            ),
            deployed=lambda p, ids: deployed_with_params(p, ids)[:2],
            checkpoint_tasks=[str(r["task"]) for r in checkpoint_rows],
            checkpoint_params=[r["params"] for r in checkpoint_rows],
            test_sets=test_sets,
            probe_sets=probe_sets,
            probe_status=probe_status,
            anchor_batches=anchor_batches,
            anchor_reason=anchor_reason,
            long_range_sets=long_sets,
            long_range_reason=long_reason,
            n_buckets=int((settings.get("long_range_proxy") or {}).get("buckets", 4)),
        )
    except Exception as exc:
        return {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"[:500]}


def _attach_external_joint_gap(records: list[dict[str, Any]]) -> None:
    """Per-domain and guarded-relative sequential-vs-joint gap (only for budget-matched pairs)."""
    from metrics.performance import sequential_joint_gap

    def final(rec: dict[str, Any], key: str) -> dict[str, float] | None:
        ext = (rec.get("deployed_evaluation") or {}).get("external") or {}
        return (ext.get("final") or {}).get(key)

    joint = {
        (
            r["method"],
            r["seed"],
            r["rank"],
            r.get("k"),
            tuple(r.get("task_order", ())),
        ): r
        for r in records
        if r.get("protocol") == "joint" and r.get("status") != "invalid"
    }
    for rec in records:
        if (
            rec.get("protocol") != "sequential"
            or rec.get("status") == "invalid"
            or rec.get("heldout_tasks")
        ):
            continue
        mate = joint.get(
            (
                rec["method"],
                rec["seed"],
                rec["rank"],
                rec.get("k"),
                tuple(rec.get("task_order", ())),
            )
        )
        if mate is None:
            continue
        if rec.get("steps") != mate.get("steps") or rec.get("tokens") != mate.get(
            "tokens"
        ):
            rec["external_joint_gap"] = {"status": "invalid_budget_mismatch"}
            continue
        gap: dict[str, Any] = {"status": "measured", "joint_run_id": mate["run_id"]}
        ext = (rec.get("deployed_evaluation") or {}).get("external") or {}
        order = ext.get("domains") or []
        base = ext.get("base_loss") or []
        for key, higher in (("loss", False), ("task_accuracy", True)):
            seq, jnt = final(rec, key), final(mate, key)
            if not seq or not jnt:
                continue
            common = [
                d
                for d in seq
                if d in jnt and np.isfinite(seq[d]) and np.isfinite(jnt[d])
            ]
            if not common:
                continue
            base_vals = (
                [base[order.index(d)] for d in common]
                if key == "loss"
                and all(d in order for d in common)
                and len(base) == len(order)
                else None
            )
            result = sequential_joint_gap(
                [seq[d] for d in common],
                [jnt[d] for d in common],
                base_vals,
                higher_is_better=higher,
            )
            result["domains"] = common
            gap[key] = result
        rec["external_joint_gap"] = (
            gap
            if len(gap) > 2
            else {"status": "unavailable", "reason": "no comparable deployed metrics"}
        )


def _load_deployed_metrics(
    run: dict[str, Any], result: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate the final checkpoint with the common deployed function.

    Heavy JAX/model imports are intentionally local so ``plan`` remains usable
    in lightweight environments.
    """
    import jax
    import jax.numpy as jnp

    from benchmark.evaluation import aggregate_lm_rows, evaluate_long_range
    from frozenllm.substrate import FrozenSubstrate, identity_modify
    from metrics.performance import (
        calibration_stats,
        ece_from_stats,
        kl_to_base,
        token_accuracy,
    )
    from metrics.representation import (
        effective_rank,
        gradient_conflict,
        invariant_redundancy,
    )
    from residual import ResidualAdapter, ResidualConfig
    from runtime.data_loader import (
        load_runtime_config,
        load_runtime_data,
        load_split_texts,
        make_eval_set,
        tokenize_texts,
    )
    from stages.results import load_params
    from stages.stage_base import TrainingConfig

    if run.get("method") not in DEFAULT_CONFIGS:
        return {"status": "unavailable", "reason": "unsupported Stage A method"}

    config_path = Path(run.get("materialized_config", run["config"]))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    training = TrainingConfig.from_dict(config.get("training") or {})
    checkpoint_dir = Path(run["output_dir"]) / run["run_name"] / "params"
    checkpoint_paths = sorted(checkpoint_dir.glob("task_*.npz"))
    if not checkpoint_paths:
        return {"status": "unavailable", "reason": "no checkpoint found"}
    substrate = FrozenSubstrate(training.model)
    adapter = ResidualAdapter(
        ResidualConfig.from_dict(config["adapter"]), substrate.architecture
    )

    def load_adapter_params(path: Path) -> dict[str, dict[str, jax.Array]]:
        loaded = _filter_adapter_params(load_params(path))
        return {
            layer: {name: jnp.asarray(value) for name, value in values.items()}
            for layer, values in loaded.items()
        }

    checkpoint_rows = []
    for path in checkpoint_paths:
        fields = path.stem.split("_", 2)
        if len(fields) != 3 or not fields[1].isdigit():
            continue
        checkpoint_rows.append(
            {
                "index": int(fields[1]),
                "task": fields[2],
                "path": path,
                "params": load_adapter_params(path),
            }
        )
    checkpoint_rows.sort(key=lambda row: int(row["index"]))
    if not checkpoint_rows:
        return {"status": "unavailable", "reason": "no valid task checkpoint found"}
    params = checkpoint_rows[-1]["params"]
    adapter.validate_params(params)
    runtime_config_path = Path(
        run.get("materialized_runtime_config", DEFAULT_RUNTIME_CONFIG)
    )
    runtime_data = load_runtime_data(
        config_path=runtime_config_path,
        tokenizer=substrate.tokenizer,
        source=run.get("runtime_source", BENCHMARK_RUNTIME_SOURCE),
        sequence_length=training.seq_len,
        batch_size=training.batch_size,
        eval_batches=training.eval_batches,
        default_epochs=training.epochs,
    )
    runtime_config = load_runtime_config(runtime_config_path)
    evaluation_config = runtime_config.get("evaluation", {})
    calibration_bins = int(evaluation_config.get("calibration_bins", 15))
    domains = runtime_data.tasks
    test_sets = runtime_data.get_test_sets(
        training.batch_size,
        training.eval_batches,
    )
    metrics_by_domain: dict[str, dict[str, Any]] = {}
    batches_by_domain: dict[str, tuple[jax.Array, ...]] = {}
    latency: list[float] = []

    def deployed_with_params(
        deployed_params: dict[str, dict[str, jax.Array]],
        ids: jax.Array,
    ) -> tuple[jax.Array, jax.Array, Any]:
        if run["method"] != "a3":
            base_result = substrate.run_with_interception(
                ids, modify_fn=identity_modify, intercept_layers=adapter.layers
            )
            adapted_result = substrate.run_with_interception(
                ids,
                modify_fn=adapter.modify_fn(deployed_params),
                intercept_layers=adapter.layers,
            )
            return base_result.logits, adapted_result.logits, adapted_result
        clean = substrate.run_with_interception(
            ids, modify_fn=identity_modify, intercept_layers=adapter.layers
        )
        latent = {
            layer: adapter.encode(
                deployed_params[f"layer_{layer}"], clean.hidden_state(layer)
            )
            for layer in adapter.layers
        }

        def modify(hidden: jax.Array, layer: int) -> jax.Array:
            block = deployed_params.get(f"layer_{layer}")
            if block is None:
                return hidden
            return hidden + adapter.scale * (latent[layer] @ block["B"]).astype(
                hidden.dtype
            )

        adapted_result = substrate.run_with_interception(
            ids, modify_fn=modify, intercept_layers=adapter.layers
        )
        return clean.logits, adapted_result.logits, adapted_result

    def deployed(ids: jax.Array) -> tuple[jax.Array, jax.Array, Any]:
        return deployed_with_params(params, ids)

    def loss_for(p: dict[str, dict[str, jax.Array]], ids: jax.Array) -> jax.Array:
        if run["method"] != "a3":
            adapted = substrate.run_with_interception(
                ids, modify_fn=adapter.modify_fn(p), intercept_layers=adapter.layers
            ).logits
        else:
            clean = substrate.run_with_interception(
                ids, modify_fn=identity_modify, intercept_layers=adapter.layers
            )
            latent = {
                layer: adapter.encode(p[f"layer_{layer}"], clean.hidden_state(layer))
                for layer in adapter.layers
            }

            def modify(hidden: jax.Array, layer: int) -> jax.Array:
                block = p.get(f"layer_{layer}")
                if block is None:
                    return hidden
                return hidden + adapter.scale * (latent[layer] @ block["B"]).astype(
                    hidden.dtype
                )

            adapted = substrate.run_with_interception(
                ids, modify_fn=modify, intercept_layers=adapter.layers
            ).logits
        return substrate.compute_loss(adapted, ids, losses=("CE",))

    for domain in domains:
        batches = test_sets[domain.name]
        batches_by_domain[domain.name] = batches
        rows: list[dict[str, float | int]] = []
        conf_sums = []
        acc_sums = []
        count_sums = []
        if batches:
            _, warmed, _ = deployed(batches[0])
            jax.block_until_ready(warmed)
        for ids in batches:
            if run["method"] == "a3":
                start = time.perf_counter()
                base, adapted, _ = deployed(ids)
            else:
                base = substrate.run_with_interception(
                    ids,
                    modify_fn=identity_modify,
                    intercept_layers=adapter.layers,
                ).logits
                jax.block_until_ready(base)
                start = time.perf_counter()
                adapted = substrate.run_with_interception(
                    ids,
                    modify_fn=adapter.modify_fn(params),
                    intercept_layers=adapter.layers,
                ).logits
            jax.block_until_ready(adapted)
            latency.append(time.perf_counter() - start)
            c_sum, a_sum, cnt = calibration_stats(adapted, ids, bins=calibration_bins)
            conf_sums.append(c_sum)
            acc_sums.append(a_sum)
            count_sums.append(cnt)
            token_count = int(np.asarray(ids[:, 1:] != -100).sum())
            adapted_loss = float(substrate.compute_loss(adapted, ids, losses=("CE",)))
            base_loss = float(substrate.compute_loss(base, ids, losses=("CE",)))
            rows.append(
                {
                    "tokens": token_count,
                    "loss_sum": adapted_loss * token_count,
                    "base_loss_sum": base_loss * token_count,
                    "kl_sum": float(kl_to_base(base, adapted)) * token_count,
                    "correct": float(token_accuracy(adapted, ids)) * token_count,
                }
            )
        if conf_sums:
            pooled_conf = jnp.sum(jnp.stack(conf_sums), axis=0)
            pooled_acc = jnp.sum(jnp.stack(acc_sums), axis=0)
            pooled_cnt = jnp.sum(jnp.stack(count_sums), axis=0)
            pooled_ece = float(ece_from_stats(pooled_conf, pooled_acc, pooled_cnt))
        else:
            pooled_ece = None
        aggregate = aggregate_lm_rows(rows)
        metrics_by_domain[domain.name] = {
            "status": aggregate["status"],
            "loss": aggregate.get("loss"),
            "ppl": aggregate.get("perplexity"),
            "token_accuracy": aggregate.get("token_accuracy"),
            "ece": pooled_ece,
            "kl": aggregate.get("kl"),
        }

    gradients: dict[str, Any] = {}
    for name, batches in batches_by_domain.items():
        if not batches:
            continue
        gradients[name] = jax.grad(
            lambda p, task_batches=batches: jnp.mean(
                jnp.stack([loss_for(p, ids) for ids in task_batches])
            )
        )(params)
    conflict_pairs = {}
    names = list(gradients)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            conflict_pairs[f"{left}:{right}"] = float(
                gradient_conflict(gradients[left], gradients[right])
            )

    representation: dict[str, Any] = {
        "effective_rank": {},
        "invariant_redundancy": {},
        "task_gradient_conflict": {
            "status": "measured" if conflict_pairs else "insufficient_data",
            "estimator": "gradient_of_mean_ce_over_configured_evaluation_batches",
            "evaluation_batches": {
                name: len(batches_by_domain[name]) for name in names
            },
            "pairs": conflict_pairs,
            "mean": (
                float(np.mean(list(conflict_pairs.values())))
                if conflict_pairs
                else None
            ),
            "max": max(conflict_pairs.values()) if conflict_pairs else None,
        },
    }
    for layer_name, block in params.items():
        if not layer_name.startswith("layer_"):
            continue
        a, b = block["A"], block["B"]
        representation["effective_rank"][layer_name] = float(effective_rank(a @ b))
        representation["invariant_redundancy"][layer_name] = float(
            invariant_redundancy(a, b)
        )

    def evaluate_lm_batches(
        evaluation_params: dict[str, dict[str, jax.Array]],
        batches: tuple[jax.Array, ...],
    ) -> dict[str, Any]:
        rows = []
        for ids in batches:
            base_logits, adapted_logits, _ = deployed_with_params(
                evaluation_params, ids
            )
            token_count = int(np.asarray(ids[:, 1:] != -100).sum())
            adapted_loss = float(
                substrate.compute_loss(adapted_logits, ids, losses=("CE",))
            )
            base_loss = float(substrate.compute_loss(base_logits, ids, losses=("CE",)))
            rows.append(
                {
                    "tokens": token_count,
                    "loss_sum": adapted_loss * token_count,
                    "base_loss_sum": base_loss * token_count,
                    "kl_sum": float(kl_to_base(base_logits, adapted_logits))
                    * token_count,
                    "correct": float(token_accuracy(adapted_logits, ids)) * token_count,
                }
            )
        return aggregate_lm_rows(rows)

    anchor_config = evaluation_config.get("anchor")
    anchor_trajectory = []
    if isinstance(anchor_config, dict) and anchor_config.get("texts"):
        anchor_seq_len = min(32, training.seq_len)
        try:
            anchor_blocks = tokenize_texts(
                [str(text) for text in anchor_config["texts"]],
                substrate.tokenizer,
                sequence_length=anchor_seq_len,
                separator="\n\n",
                task_name=str(anchor_config.get("name", "anchor")),
            )
            anchor_batches = make_eval_set(
                anchor_blocks,
                min(training.batch_size, len(anchor_blocks)),
                training.eval_batches,
            )
            for checkpoint in checkpoint_rows:
                anchor_trajectory.append(
                    {
                        "checkpoint": int(checkpoint["index"]),
                        **evaluate_lm_batches(checkpoint["params"], anchor_batches),
                    }
                )
            anchor_kl: dict[str, Any] = {
                "status": "measured",
                "dataset": str(anchor_config.get("name", "anchor")),
                "trajectory": anchor_trajectory,
                "final": anchor_trajectory[-1],
            }
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            anchor_kl = {"status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    else:
        anchor_kl = {
            "status": "not_configured",
            "reason": "anchor dataset not configured",
        }

    heldout_transfer: dict[str, Any] = {}
    for dataset_config in runtime_config.get("datasets", []):
        if dataset_config.get("type") != "heldout":
            continue
        name = str(dataset_config["name"])
        try:
            texts = load_split_texts(dataset_config, dataset_config["test_split"])
            blocks = tokenize_texts(
                texts,
                substrate.tokenizer,
                sequence_length=training.seq_len,
                separator=str(dataset_config.get("separator", "\n\n")),
                task_name=name,
            )
            batches = make_eval_set(
                blocks,
                training.batch_size,
                training.eval_batches,
            )
            trajectory = [
                {
                    "checkpoint": int(checkpoint["index"]),
                    **evaluate_lm_batches(checkpoint["params"], batches),
                }
                for checkpoint in checkpoint_rows
            ]
            final = trajectory[-1]
            heldout_transfer[name] = {
                "status": "measured",
                "trajectory": trajectory,
                "loss_transfer": float(final["base_loss"]) - float(final["loss"]),
            }
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            heldout_transfer[name] = {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }

    long_range_config = evaluation_config.get("long_range")
    if isinstance(long_range_config, dict):

        def score_choice(prompt: str, choice: str) -> float:
            prompt_ids = substrate.tokenizer.encode(prompt, add_special_tokens=True)
            candidate = substrate.tokenizer.encode(choice, add_special_tokens=False)
            if not candidate:
                return float("-inf")
            full_ids = prompt_ids + candidate
            max_positions = substrate.architecture.max_position_embeddings
            if max_positions is not None and len(full_ids) > max_positions:
                raise ValueError("long-range probe exceeds model context length")
            ids = jnp.asarray([full_ids], dtype=jnp.int32)
            _, adapted_logits, _ = deployed_with_params(params, ids)
            log_probs = jax.nn.log_softmax(adapted_logits[0], axis=-1)
            start = len(prompt_ids) - 1
            values = [
                log_probs[start + offset, token]
                for offset, token in enumerate(candidate)
            ]
            return float(jnp.mean(jnp.stack(values)))

        try:
            long_range = evaluate_long_range(long_range_config, score_choice)
        except (RuntimeError, TypeError, ValueError) as exc:
            long_range = {
                "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}",
            }
    else:
        long_range = {
            "status": "not_configured",
            "reason": "long-range probes are not configured",
        }
    # External metrics (see the external-metrics block above): per-checkpoint evaluation.
    external = _external_block(
        substrate=substrate,
        deployed_with_params=deployed_with_params,
        checkpoint_rows=checkpoint_rows,
        test_sets=test_sets,
        training=training,
        runtime_config_path=runtime_config_path,
    )
    peak_bytes = None
    for device in jax.devices():
        stats = device.memory_stats()
        value = stats.get("peak_bytes_in_use") if stats else None
        if value is not None:
            peak_bytes = max(peak_bytes or 0, int(value))
    return {
        "status": "measured",
        "domains": metrics_by_domain,
        "representation": representation,
        "external": external,
        "ops": {
            "inference_latency_mean_s": float(np.mean(latency)) if latency else None,
            "inference_latency_p95_s": (
                float(np.percentile(latency, 95)) if latency else None
            ),
            "peak_memory_bytes": peak_bytes,
        },
        "anchor_kl": anchor_kl,
        "heldout_transfer": heldout_transfer
        or {"status": "not_configured", "reason": "held-out datasets not configured"},
        "long_range_coherence": long_range,
    }


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _config_fingerprint(config: dict[str, Any]) -> str:
    return _sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _filter_adapter_params(params: dict[str, Any]) -> dict[str, Any]:
    """Keep only deployable residual-adapter parameter blocks."""
    return {name: block for name, block in params.items() if name.startswith("layer_")}


def _compressibility_curve(deltas: dict[int, float]) -> dict[int, float]:

    return _compressibility_detail(deltas)["curve"] or {}


def _compressibility_detail(deltas: dict[int, float]) -> dict[str, Any]:
    from metrics.representation import guarded_compressibility

    return guarded_compressibility(deltas)


def _set_seed(config: dict[str, Any], method: str, seed: int) -> None:
    config.setdefault("adapter", {})["seed"] = seed
    config.setdefault("training", {})["seed"] = seed
    if method == "a2":
        config.setdefault("objective", {})["seed"] = seed
        hid = config["objective"].setdefault("hid", {})
        if not isinstance(hid, dict):
            raise ValueError("objective.hid must be a mapping")
        predictor = hid.setdefault("predictor", {})
        if not isinstance(predictor, dict):
            raise ValueError("objective.hid.predictor must be a mapping")
        predictor["seed"] = seed


def _set_epochs(config: dict[str, Any], epochs: int | None) -> None:
    if epochs is None:
        return
    config.setdefault("training", {})["epochs"] = epochs
    for domain in config.get("domains", []):
        if isinstance(domain, dict):
            domain["epochs"] = epochs


def _set_protocol(config: dict[str, Any], protocol: str) -> None:
    if protocol not in {"sequential", "joint", "heldout", "commutator"}:
        raise ValueError(f"Unsupported protocol: {protocol!r}")


def _set_rank(config: dict[str, Any], rank: int) -> None:
    if rank < 1:
        raise ValueError("rank must be >= 1")
    config.setdefault("adapter", {})["rank"] = rank


def _set_shared_training(config: dict[str, Any]) -> None:
    training = config.setdefault("training", {})
    training["learning_rate"] = BENCHMARK_LEARNING_RATE
    training["reset_optimizer"] = False


def _set_method_variant(config: dict[str, Any], spec: RunSpec) -> None:
    if spec.method == "a2" and spec.a2_arm == "full":
        objective = config.setdefault("objective", {})
        objective["cpc"] = {
            **objective.get("cpc", {}),
            "weight": 0.05,
            "positive": "base",
        }
        objective["stab"] = {
            **objective.get("stab", {}),
            "weight": 0.1,
            "w_anch": 1.0,
            "w_smooth": 0.0,
        }
        hid = {**objective.get("hid", {}), "weight": 0.1}
        hid["predictor"] = {**hid.get("predictor", {}), "detach_input": False}
        objective["hid"] = hid
        objective["hid_detach_input"] = False
    if spec.method == "a3" and spec.k is not None:
        config.setdefault("a3_hybrid", {})["infer_steps"] = spec.k


def _validate_domains(
    config: dict[str, Any],
    source: Path,
    runtime_source: str = BENCHMARK_RUNTIME_SOURCE,
) -> None:
    datasets = config.get("datasets")
    names = (
        tuple(
            dataset.get("name")
            for dataset in datasets
            if isinstance(dataset, dict) and dataset.get("type") == runtime_source
        )
        if isinstance(datasets, list)
        else ()
    )
    if names != EXPECTED_DOMAINS:
        raise ValueError(
            f"{source} must define {runtime_source!r} datasets in order "
            f"{list(EXPECTED_DOMAINS)}, "
            f"got {list(names)}"
        )


def expand_specs(
    output_dir: Path,
    *,
    seeds: list[int],
    methods: list[str],
    k_values: list[int],
    rank_values: list[int] | None = None,
    epochs: int | None = None,
    protocol: str = "sequential",
    task_order: list[str] | tuple[str, ...] | None = None,
    heldout_tasks: list[str] | tuple[str, ...] | None = None,
) -> list[RunSpec]:
    if not seeds:
        raise ValueError("At least one seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique")
    unknown = sorted(set(methods) - set(DEFAULT_CONFIGS))
    if unknown:
        raise ValueError(
            f"Unknown methods: {unknown}; choose from {sorted(DEFAULT_CONFIGS)}"
        )
    if not k_values or any(k < 0 for k in k_values):
        raise ValueError("k_values must contain non-negative integers.")
    if len(set(k_values)) != len(k_values):
        raise ValueError("k_values must be unique")
    ranks = rank_values or [16]
    if not ranks or any(rank < 1 for rank in ranks):
        raise ValueError("rank_values must contain positive integers.")
    if len(set(ranks)) != len(ranks):
        raise ValueError("rank_values must be unique")
    specs: list[RunSpec] = []
    order = tuple(task_order or EXPECTED_DOMAINS)
    heldout = tuple(heldout_tasks or ())
    if set(order) != set(EXPECTED_DOMAINS) or len(order) != len(EXPECTED_DOMAINS):
        raise ValueError(
            f"task_order must contain {list(EXPECTED_DOMAINS)} exactly once"
        )
    if any(name not in EXPECTED_DOMAINS for name in heldout):
        raise ValueError(
            f"heldout_tasks must be selected from {list(EXPECTED_DOMAINS)}"
        )
    for method in methods:
        for seed in seeds:
            ks: list[int | None] = (
                [int(k) for k in k_values] if method == "a3" else [None]
            )
            for rank in ranks:
                for k in ks:
                    suffix = f"-r{rank}" + (f"-k{k}" if k is not None else "")
                    order_tag = "-".join(order)
                    heldout_tag = "" if not heldout else f"-holdout-{'-'.join(heldout)}"
                    run_id = f"{method}-seed{seed}{suffix}-{protocol}-{order_tag}{heldout_tag}"
                    attempts = output_dir / "runs" / run_id / "attempts"
                    specs.append(
                        RunSpec(
                            method=method,
                            seed=seed,
                            protocol=protocol,
                            run_id=run_id,
                            config=str(DEFAULT_CONFIGS[method]),
                            output_dir=str(attempts),
                            run_name="attempt-0001",
                            k=k,
                            rank=rank,
                            runtime_config=str(DEFAULT_RUNTIME_CONFIG),
                            task_order=order,
                            heldout_tasks=heldout,
                            a2_arm="full" if method == "a2" else None,
                        )
                    )
    return specs


def _materialize(spec: RunSpec, root: Path, epochs: int | None) -> Path:
    source = Path(spec.config)
    config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    runtime_source = Path(spec.runtime_config)
    runtime_config = yaml.safe_load(runtime_source.read_text(encoding="utf-8")) or {}
    _validate_domains(runtime_config, runtime_source, spec.runtime_source)
    _set_seed(config, spec.method, spec.seed)
    _set_rank(config, spec.rank)
    _set_shared_training(config)
    _set_epochs(config, epochs)
    _set_protocol(config, spec.protocol)
    _set_method_variant(config, spec)
    target = root / "configs" / f"{spec.run_id}.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return target


def _materialize_runtime(spec: RunSpec, root: Path) -> Path:
    source = Path(spec.runtime_config)
    config = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    config.setdefault("runtime", {})["source"] = spec.runtime_source
    config["runtime"]["training_mode"] = (
        "joint" if spec.protocol == "joint" else "sequential"
    )
    config["runtime"]["task_order"] = list(spec.task_order)
    config["runtime"]["heldout_tasks"] = list(spec.heldout_tasks)
    target = root / "configs" / f"{spec.run_id}.runtime.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return target


def _condition(
    spec: RunSpec,
    config: Path,
    runtime_config: Path,
    epochs: int | None,
    source_sha256: str,
) -> dict[str, Any]:
    return effective_condition(
        method=spec.method,
        seed=spec.seed,
        rank=spec.rank,
        k=spec.k,
        protocol=spec.protocol,
        task_order=spec.task_order,
        heldout_tasks=spec.heldout_tasks,
        a2_arm=spec.a2_arm,
        stage_config_sha256=sha256_file(config),
        runtime_config_sha256=sha256_file(runtime_config),
        runtime_source=spec.runtime_source,
        epochs=epochs,
        source_sha256=source_sha256,
    )


def make_manifest(
    output_dir: Path,
    *,
    seeds: list[int],
    methods: list[str],
    k_values: list[int],
    rank_values: list[int] | None = None,
    epochs: int | None = None,
    protocol: str = "sequential",
    protocols: list[str] | tuple[str, ...] | None = None,
    task_order: list[str] | tuple[str, ...] | None = None,
    heldout_tasks: list[str] | tuple[str, ...] | None = None,
    include_a4_plan: bool = False,
    profile: str = "custom",
) -> dict[str, Any]:
    selected_protocols = tuple(protocols or (protocol,))
    if len(set(selected_protocols)) != len(selected_protocols):
        raise ValueError("protocols must be unique")
    specs = []
    base_order = tuple(task_order or EXPECTED_DOMAINS)
    for selected in selected_protocols:
        if selected == "commutator":
            orders = (base_order, tuple(reversed(base_order)))
            heldout_variants = ((),)
        elif selected == "heldout" and not heldout_tasks:
            orders = (base_order,)
            heldout_variants = tuple((name,) for name in EXPECTED_DOMAINS)
        else:
            orders = (base_order,)
            heldout_variants = (tuple(heldout_tasks or ()),)
        for order in orders:
            for heldout in heldout_variants:
                specs.extend(
                    expand_specs(
                        output_dir,
                        seeds=seeds,
                        methods=methods,
                        k_values=k_values,
                        rank_values=rank_values,
                        epochs=epochs,
                        protocol=selected,
                        task_order=order,
                        heldout_tasks=heldout,
                    )
                )
    source = source_fingerprint(ROOT)
    run_rows = []
    for spec in specs:
        config = _materialize(spec, output_dir, epochs)
        runtime_config = _materialize_runtime(spec, output_dir)
        condition = _condition(
            spec, config, runtime_config, epochs, str(source["source_sha256"])
        )
        run_rows.append(
            {
                **asdict(spec),
                "status": "pending",
                "materialized_config": str(config),
                "materialized_runtime_config": str(runtime_config),
                **condition,
            }
        )
    a4_runs = []
    if include_a4_plan:
        parent_rows = {row["run_id"]: row for row in run_rows}
        a4_runs = [
            {
                "schema": "nsllm.a4-child/v1",
                "run_id": f"a4-{spec.run_id.removeprefix('a3-')}",
                "method": "a4",
                "parent_a3_run_id": spec.run_id,
                "parent_a3_condition_sha256": parent_rows[spec.run_id][
                    "condition_sha256"
                ],
                "seed": spec.seed,
                "rank": spec.rank,
                "k": spec.k,
                "protocol": spec.protocol,
                "task_order": list(spec.task_order),
                "heldout_tasks": list(spec.heldout_tasks),
                "parent_a3_checkpoint_path": None,
                "parent_a3_checkpoint_sha256": None,
                "parent_a3_model": None,
                "parent_a3_layers": None,
                "parent_a3_scale": None,
                "parent_a3_activation": None,
                "a3_parent_schema": "nsllm.a3-parent/v1",
                "symbolic_bank_sha256": None,
                "alignment_manifest_sha256": None,
                "retrieval_probes_sha256": None,
                "native_library_sha256": None,
                "symbolic_dim": None,
                "top_m": None,
                "retrieval_backend": None,
                "required_artifacts": [
                    "parent_a3_checkpoint",
                    "symbolic_bank",
                    "alignment_manifest",
                    "retrieval_probes",
                    "retrieval_events",
                    "bank_snapshots",
                    "consolidation_events",
                    "measurement_artifacts",
                ],
                "status": "requires_a4_implementation",
                "symbolic_metrics": list(A4_SYMBOLIC_METRICS),
                "deferred_later_stage_metrics": list(LATER_STAGE_SYMBOLIC_METRICS),
            }
            for spec in specs
            if spec.method == "a3"
        ]
    manifest = {
        "schema": 2,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "repository": str(ROOT),
        "protocols": list(selected_protocols),
        "profile": profile,
        "methods": methods,
        "seeds": seeds,
        "k_values": k_values,
        "rank_values": rank_values or [16],
        "epochs": epochs,
        "domains": list(EXPECTED_DOMAINS),
        "runtime_config": str(DEFAULT_RUNTIME_CONFIG),
        "runtime_source": BENCHMARK_RUNTIME_SOURCE,
        "source": source,
        "measured_metrics": list(MEASURED_METRICS),
        "runs": run_rows,
        "a4_runs": a4_runs,
    }
    _write_json(output_dir / "manifest.json", manifest)
    return manifest


def _update_manifest(path: Path, run_id: str, **fields: Any) -> None:
    manifest = _read_json(path)
    for run in manifest["runs"]:
        if run["run_id"] == run_id:
            run.update(fields)
            break
    else:
        raise KeyError(run_id)
    _write_json(path, manifest)


def _command(
    spec: RunSpec, config: Path, runtime_config: Path | None = None
) -> list[str]:
    runtime_config = runtime_config or Path(spec.runtime_config)
    command = [
        sys.executable,
        str(RUNNERS[spec.method]),
        "--config",
        str(config),
        "--output-dir",
        spec.output_dir,
        "--run-name",
        spec.run_name,
        "--runtime-source",
        spec.runtime_source,
        "--runtime-config",
        str(runtime_config),
        "--training-mode",
        "joint" if spec.protocol == "joint" else "sequential",
        "--task-order",
        *spec.task_order,
    ]
    for name in spec.heldout_tasks:
        command.extend(["--heldout-task", name])
    if spec.method == "a2" and spec.a2_arm is not None:
        command.extend(["--arm", spec.a2_arm])
    return command


def train_manifest(
    output_dir: Path, *, resume: bool = True, epochs: int | None = None
) -> None:
    manifest_path = output_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    config_root = output_dir
    planned_source_sha = str(manifest.get("source", {}).get("source_sha256", ""))
    for raw in manifest["runs"]:
        current_source = source_fingerprint(ROOT)
        if planned_source_sha and current_source["source_sha256"] != planned_source_sha:
            raise RuntimeError(
                "Repository source changed after planning; create a new manifest before training."
            )
        spec = RunSpec(
            method=raw["method"],
            seed=raw["seed"],
            protocol=raw["protocol"],
            run_id=raw["run_id"],
            config=raw["config"],
            output_dir=raw["output_dir"],
            run_name=raw["run_name"],
            k=raw.get("k"),
            rank=raw.get("rank", 16),
            runtime_source=raw.get("runtime_source", BENCHMARK_RUNTIME_SOURCE),
            runtime_config=raw.get("runtime_config", str(DEFAULT_RUNTIME_CONFIG)),
            task_order=tuple(raw.get("task_order", EXPECTED_DOMAINS)),
            heldout_tasks=tuple(raw.get("heldout_tasks", ())),
            a2_arm=raw.get("a2_arm", "full" if raw["method"] == "a2" else None),
        )
        effective_epochs = epochs if epochs is not None else manifest.get("epochs")
        config = _materialize(
            spec,
            config_root,
            effective_epochs,
        )
        runtime_config = _materialize_runtime(spec, config_root)
        condition = _condition(
            spec,
            config,
            runtime_config,
            effective_epochs,
            str(current_source["source_sha256"]),
        )
        config_sha = sha256_file(config)
        previous_run = Path(raw["run_dir"]) if raw.get("run_dir") else None
        if resume and previous_run is not None:
            reusable, reason = validate_complete_attempt(
                previous_run,
                condition_sha256=condition["condition_sha256"],
                config_sha256=config_sha,
            )
            if reusable:
                _update_manifest(
                    manifest_path,
                    spec.run_id,
                    status="complete",
                    results=str(previous_run / "results.json"),
                    resume_validation="verified",
                )
                continue
            raw["resume_rejection"] = reason

        attempts_root = Path(spec.output_dir)
        attempt_name = next_attempt_name(attempts_root)
        attempt_spec = replace(spec, run_name=attempt_name)
        run_dir = attempts_root / attempt_name
        results = run_dir / "results.json"
        _update_manifest(
            manifest_path,
            spec.run_id,
            status="running",
            materialized_config=str(config),
            materialized_runtime_config=str(runtime_config),
            config_sha=config_sha,
            run_name=attempt_name,
            run_dir=str(run_dir),
            **condition,
            started_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )
        log_path = output_dir / "logs" / f"{spec.run_id}-{attempt_name}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                _command(attempt_spec, config, runtime_config),
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
        if run_dir.is_dir():
            run_log = run_dir / "logs" / "stdout.log"
            run_log.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(log_path, run_log)
        if completed.returncode != 0:
            internal_log = run_dir / "logs" / "stdout.log"
            _update_manifest(
                manifest_path,
                spec.run_id,
                status="failed",
                returncode=completed.returncode,
                log=str(internal_log if internal_log.is_file() else log_path),
                run_dir=str(run_dir),
            )
            try:
                log_tail = "\n".join(
                    log_path.read_text(encoding="utf-8", errors="replace").splitlines()[
                        -40:
                    ]
                )
            except OSError as exc:
                log_tail = f"Unable to read failure log: {exc}"
            raise RuntimeError(
                f"{spec.run_id} failed with exit code {completed.returncode}; "
                f"see {log_path}\n\nLast log lines:\n{log_tail}"
            )
        if not results.is_file():
            raise RuntimeError(
                f"{spec.run_id} exited successfully but wrote no results.json"
            )
        provenance_dir = run_dir / "provenance"
        provenance_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config, provenance_dir / "materialized_stage.yaml")
        shutil.copy2(runtime_config, provenance_dir / "materialized_runtime.yaml")
        completion_spec = {
            **condition,
            "run_id": spec.run_id,
            "attempt": attempt_name,
            "config_sha256": config_sha,
            "runtime_config_sha256": sha256_file(runtime_config),
            "source": current_source,
        }
        finalize_attempt(run_dir, completion_spec)
        _update_manifest(
            manifest_path,
            spec.run_id,
            status="complete",
            returncode=0,
            results=str(results),
            log=str(run_dir / "logs" / "stdout.log"),
            run_dir=str(run_dir),
            finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        )


def _result_record(raw: dict[str, Any], run: dict[str, Any]) -> dict[str, Any]:
    test = raw.get("test", {})
    domains = raw.get("domains", [])
    losses = raw.get("loss", [])
    final_losses = losses[-1] if losses else []
    perplexities = raw.get("ppl", [])
    base_losses = raw.get("base_loss", [])
    final_mean_loss = sum(final_losses) / len(final_losses) if final_losses else None
    base_mean_loss = sum(base_losses) / len(base_losses) if base_losses else None
    if run.get("protocol") == "joint":
        acquisition_losses = list(final_losses)
        acquisition_improvement = {
            str(name): float(base) - float(loss)
            for name, base, loss in zip(
                domains, base_losses, final_losses, strict=False
            )
        }
    else:
        acquisition_losses = [
            losses[index][index]
            for index in range(min(len(domains), len(losses)))
            if index < len(losses[index])
        ]
        acquisition_improvement = (
            {
                str(domains[index]): float(base_losses[index])
                - float(losses[index][index])
                for index in range(min(len(domains), len(base_losses), len(losses)))
                if index < len(losses[index])
            }
            if domains
            else None
        )
    return {
        "run_id": run["run_id"],
        "method": run["method"],
        "seed": run["seed"],
        "k": run.get("k"),
        "rank": run.get("rank", 16),
        "protocol": run.get("protocol", "sequential"),
        "task_order": run.get("task_order", list(EXPECTED_DOMAINS)),
        "heldout_tasks": run.get("heldout_tasks", []),
        "condition_sha256": run.get("condition_sha256"),
        "domains": domains,
        "final_mean_loss": final_mean_loss,
        "domain_loss_improvement": acquisition_improvement,
        "loss_improvement": (
            base_mean_loss - (sum(acquisition_losses) / len(acquisition_losses))
            if base_mean_loss is not None and acquisition_losses
            else None
        ),
        "final_mean_ppl": (
            math.exp(final_mean_loss) if final_mean_loss is not None else None
        ),
        "final_test_mean_loss": (
            sum(float(row["loss"]) for row in test.values()) / len(test)
            if test
            else None
        ),
        "final_test_mean_ppl": (
            math.exp(sum(float(row["loss"]) for row in test.values()) / len(test))
            if test
            else None
        ),
        "loss_matrix": losses,
        "perplexity_matrix": perplexities,
        "mean_forgetting": (
            raw.get("mean_forgetting", [])[-1] if raw.get("mean_forgetting") else None
        ),
        "steps": raw.get("steps"),
        "tokens": raw.get("tokens"),
        "mean_step_time": raw.get("mean_step_time"),
        "params_unchanged": raw.get("params_unchanged"),
        "results": run.get("results"),
        "heldout_transfer": {
            name: float(test[name]["base_loss"]) - float(test[name]["loss"])
            for name in run.get("heldout_tasks", [])
            if name in test
        }
        or None,
    }


def evaluate_manifest(output_dir: Path) -> dict[str, Any]:
    manifest = _read_json(output_dir / "manifest.json")
    records = []
    fingerprints = {}
    for run in manifest["runs"]:
        result_path = Path(run.get("results", ""))
        if not result_path.is_file():
            continue
        config_path = Path(run.get("materialized_config", run["config"]))
        config_bytes = config_path.read_bytes()
        fingerprints[run["run_id"]] = _sha256(config_bytes)
        if manifest.get("schema", 1) >= 2:
            run_dir = Path(run.get("run_dir", result_path.parent))
            valid, reason = validate_complete_attempt(
                run_dir,
                condition_sha256=str(run.get("condition_sha256", "")),
                config_sha256=sha256_file(config_path),
            )
            if not valid:
                records.append(
                    {
                        "run_id": run["run_id"],
                        "method": run["method"],
                        "seed": run["seed"],
                        "rank": run.get("rank", 16),
                        "k": run.get("k"),
                        "protocol": run.get("protocol"),
                        "status": "invalid",
                        "reason": reason,
                    }
                )
                continue
        raw_result = _read_json(result_path)
        record = _result_record(raw_result, run)
        meta_path = result_path.parent / "meta.json"
        if meta_path.is_file():
            meta = _read_json(meta_path)
            record["trainable_params"] = meta.get("trainable_params")
            record["trainable_params_by_group"] = meta.get("trainable_params_by_group")
            if isinstance(meta.get("runtime_data"), dict):
                record["runtime_data_sha256"] = _config_fingerprint(
                    meta["runtime_data"]
                )
        try:
            evaluation_run = {
                **run,
                "evaluation_output_dir": str(
                    output_dir / "evaluation_artifacts" / run["run_id"]
                ),
            }
            deployed = _load_deployed_metrics(evaluation_run, raw_result)
            record["deployed_evaluation"] = deployed
            evaluation_dir = Path(evaluation_run["evaluation_output_dir"])
            if evaluation_dir.is_dir():
                record["evaluation_artifacts"] = {
                    path.relative_to(evaluation_dir).as_posix(): sha256_file(path)
                    for path in sorted(evaluation_dir.rglob("*"))
                    if path.is_file()
                }
        except (ImportError, KeyError, OSError, RuntimeError, ValueError) as exc:
            record["deployed_evaluation"] = {
                "status": "unavailable",
                "reason": f"{type(exc).__name__}: {exc}",
            }
        records.append(record)
    by_group: dict[tuple[Any, ...], dict[int, float]] = {}
    by_group_domain: dict[tuple[Any, ...], dict[str, dict[int, float]]] = {}
    for record in records:
        if record.get("status") == "invalid":
            continue
        group_key = (
            record["method"],
            record["seed"],
            record.get("k"),
            record.get("protocol"),
            tuple(record.get("task_order", ())),
            tuple(record.get("heldout_tasks", ())),
        )
        loss = record.get("loss_improvement")
        if loss is not None:
            by_group.setdefault(group_key, {})[record["rank"]] = loss
        for name, value in (record.get("domain_loss_improvement") or {}).items():
            by_group_domain.setdefault(group_key, {}).setdefault(name, {})[
                record["rank"]
            ] = value
    for record in records:
        if record.get("status") == "invalid":
            continue
        group_key = (
            record["method"],
            record["seed"],
            record.get("k"),
            record.get("protocol"),
            tuple(record.get("task_order", ())),
            tuple(record.get("heldout_tasks", ())),
        )
        detail = _compressibility_detail(by_group.get(group_key, {}))
        record["compressibility_status"] = detail["status"]
        record["compressibility_detail"] = {
            k: v for k, v in detail.items() if k != "curve"
        }
        record["residual_compressibility"] = (
            {str(rank): float(v) for rank, v in detail["curve"].items()}
            if detail["curve"]
            else None
        )
        per_domain = {}
        for name, deltas in by_group_domain.get(group_key, {}).items():
            d = _compressibility_detail(deltas)
            per_domain[name] = {
                "status": d["status"],
                "reference_improvement": d["reference_improvement"],
                "curve": (
                    {str(r): float(v) for r, v in d["curve"].items()}
                    if d["curve"]
                    else None
                ),
            }
        record["residual_compressibility_by_domain"] = per_domain or None
    reference = {
        (
            record["seed"],
            record["rank"],
            record.get("protocol"),
            tuple(record.get("task_order", ())),
            tuple(record.get("heldout_tasks", ())),
        ): record["final_mean_loss"]
        for record in records
        if record.get("method") == "a1" and record.get("status") != "invalid"
    }
    for record in records:
        if record.get("status") == "invalid":
            continue
        baseline = reference.get(
            (
                record["seed"],
                record["rank"],
                record.get("protocol"),
                tuple(record.get("task_order", ())),
                tuple(record.get("heldout_tasks", ())),
            )
        )
        record["paired_delta_vs_a1"] = (
            record["final_mean_loss"] - baseline
            if baseline is not None and record["method"] != "a1"
            else None
        )

    joint_reference = {
        (
            record["method"],
            record["seed"],
            record["rank"],
            record.get("k"),
            tuple(record.get("task_order", ())),
        ): record
        for record in records
        if record.get("protocol") == "joint" and record.get("status") != "invalid"
    }
    for record in records:
        if record.get("status") == "invalid":
            continue
        key = (
            record["method"],
            record["seed"],
            record["rank"],
            record.get("k"),
            tuple(record.get("task_order", ())),
        )
        joint_record = joint_reference.get(key)
        budgets_match = bool(
            joint_record
            and record.get("steps") == joint_record.get("steps")
            and record.get("tokens") == joint_record.get("tokens")
        )
        record["sequential_joint_gap"] = (
            record["final_mean_loss"] - joint_record["final_mean_loss"]
            if record.get("protocol") == "sequential"
            and record.get("final_mean_loss") is not None
            and joint_record is not None
            and joint_record.get("final_mean_loss") is not None
            and budgets_match
            else None
        )
        if (
            record.get("protocol") == "sequential"
            and joint_record
            and not budgets_match
        ):
            record["sequential_joint_gap_status"] = "invalid_budget_mismatch"

    _attach_external_joint_gap(records)

    commutator_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for record in records:
        if record.get("protocol") != "commutator" or record.get("status") == "invalid":
            continue
        key = (
            record["method"],
            record["seed"],
            record["rank"],
            record.get("k"),
        )
        commutator_groups.setdefault(key, []).append(record)
    for group in commutator_groups.values():
        if len(group) != 2:
            continue
        from stages.results import load_params as load_saved_params

        parameter_vectors = []
        operator_vectors = []
        for record in group:
            run = next(
                item for item in manifest["runs"] if item["run_id"] == record["run_id"]
            )
            checkpoint_dir = Path(run["output_dir"]) / run["run_name"] / "params"
            paths = sorted(checkpoint_dir.glob("task_*.npz"))
            loaded = (
                _filter_adapter_params(load_saved_params(paths[-1])) if paths else {}
            )
            parameter_vectors.append(
                np.concatenate(
                    [
                        array.ravel()
                        for _, block in sorted(loaded.items())
                        for array in (np.asarray(block["A"]), np.asarray(block["B"]))
                    ]
                )
                if loaded
                else np.asarray([])
            )
            operator_vectors.append(
                np.concatenate(
                    [
                        (np.asarray(block["A"]) @ np.asarray(block["B"])).ravel()
                        for _, block in sorted(loaded.items())
                    ]
                )
                if loaded
                else np.asarray([])
            )
        if (
            parameter_vectors[0].shape != parameter_vectors[1].shape
            or operator_vectors[0].shape != operator_vectors[1].shape
            or not parameter_vectors[0].size
            or not operator_vectors[0].size
        ):
            continue
        parameter_scale = 0.5 * (
            np.linalg.norm(parameter_vectors[0]) + np.linalg.norm(parameter_vectors[1])
        )
        operator_scale = 0.5 * (
            np.linalg.norm(operator_vectors[0]) + np.linalg.norm(operator_vectors[1])
        )
        parameter_value = float(
            np.linalg.norm(parameter_vectors[0] - parameter_vectors[1])
            / max(parameter_scale, 1e-8)
        )
        operator_value = float(
            np.linalg.norm(operator_vectors[0] - operator_vectors[1])
            / max(operator_scale, 1e-8)
        )
        for record in group:
            record["parameter_update_commutator"] = parameter_value
            record["operator_update_commutator"] = operator_value
            record["update_commutator"] = operator_value
    evaluation = {
        "schema": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "domains": manifest.get("domains", list(EXPECTED_DOMAINS)),
        "measured_metrics": manifest.get("measured_metrics", list(MEASURED_METRICS)),
        "config_fingerprints": fingerprints,
        "records": records,
        "completed_runs": sum(row.get("status") != "invalid" for row in records),
        "invalid_runs": sum(row.get("status") == "invalid" for row in records),
        "planned_runs": len(manifest["runs"]),
    }
    _write_json(output_dir / "evaluation.json", evaluation)
    return evaluation


def _numeric_values(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return [float(value)]
    if isinstance(value, dict):
        values: list[float] = []
        for item in value.values():
            values.extend(_numeric_values(item))
        return values
    if isinstance(value, list | tuple):
        values = []
        for item in value:
            values.extend(_numeric_values(item))
        return values
    return []


def _report_metric_value(record: dict[str, Any], metric: str, rank: int) -> list[float]:
    deployed = record.get("deployed_evaluation", {})
    representation = deployed.get("representation", {})
    if metric == "residual_compressibility":
        curve = record.get("residual_compressibility") or {}
        value = curve.get(str(rank))
    elif metric == "validation_loss":
        value = record.get("final_mean_loss")
    elif metric == "test_loss":
        value = record.get("final_test_mean_loss")
    elif metric == "perplexity":
        value = record.get("final_mean_ppl")
    elif metric == "test_perplexity":
        value = record.get("final_test_mean_ppl")
    elif metric == "forgetting":
        value = record.get("mean_forgetting")
    elif metric in {
        "deployed_loss",
        "deployed_perplexity",
        "token_accuracy",
        "calibration",
        "domain_kl",
    }:
        field = {
            "deployed_loss": "loss",
            "deployed_perplexity": "ppl",
            "token_accuracy": "token_accuracy",
            "calibration": "ece",
            "domain_kl": "kl",
        }[metric]
        value = {
            domain: values.get(field)
            for domain, values in deployed.get("domains", {}).items()
            if isinstance(values, dict) and values.get("status") == "measured"
        }
    elif metric == "effective_rank":
        value = representation.get("effective_rank")
    elif metric == "invariant_redundancy":
        value = representation.get("invariant_redundancy")
    elif metric == "task_gradient_conflict":
        value = representation.get("task_gradient_conflict", {}).get("mean")
    elif metric == "update_commutator":
        value = record.get(
            "operator_update_commutator", record.get("update_commutator")
        )
    elif metric == "heldout_transfer":
        values = []
        if record.get("heldout_transfer"):
            values.extend(record["heldout_transfer"].values())
        for row in deployed.get("heldout_transfer", {}).values():
            if isinstance(row, dict) and row.get("status") == "measured":
                values.append(row.get("loss_transfer"))
        value = values
    elif metric == "sequential_joint_gap":
        value = record.get("sequential_joint_gap")
    elif metric == "anchor_kl":
        anchor = deployed.get("anchor_kl", {})
        value = (
            anchor.get("final", {}).get("kl")
            if anchor.get("status") == "measured"
            else None
        )
    elif metric == "long_range":
        coherence = deployed.get("long_range_coherence", {})
        value = (
            coherence.get("accuracy") if coherence.get("status") == "measured" else None
        )
    elif metric.startswith("ext_"):
        external = deployed.get("external") or {}
        final = external.get("final") or {}
        last = lambda v: v[-1] if isinstance(v, list) and v else None  # noqa: E731
        value = {
            "ext_token_accuracy": final.get("token_accuracy"),
            "ext_task_accuracy": final.get("task_accuracy"),
            "ext_task_f1": final.get("task_f1"),
            "ext_task_micro_f1": final.get("task_micro_f1"),
            "ext_task_ece": final.get("task_ece"),
            "ext_bwt_loss": last(external.get("backward_transfer_loss")),
            "ext_bwt_accuracy": last(external.get("backward_transfer_accuracy")),
            "ext_zero_shot_gain": (
                (external.get("heldout_transfer") or {}).get("final") or {}
            ).get("gain"),
            "ext_zero_shot_ratio": (
                (external.get("heldout_transfer") or {}).get("final") or {}
            ).get("ratio"),
            "ext_unseen_gain": (external.get("unseen_transfer") or {}).get("mean_gain"),
            "ext_unseen_ratio": (external.get("unseen_transfer") or {}).get(
                "mean_ratio"
            ),
            "ext_joint_gap_relative": (
                (record.get("external_joint_gap") or {}).get("loss") or {}
            ).get("mean_relative_gap"),
            "ext_anchor_kl_final": (external.get("anchor") or {}).get("final_kl"),
            "ext_anchor_kl_peak": (external.get("anchor") or {}).get("peak_kl"),
            "ext_anchor_loss_change": (external.get("anchor") or {}).get(
                "final_loss_change"
            ),
            "ext_long_range_proxy": (external.get("long_range_proxy") or {}).get(
                "mean_delta_vs_base"
            ),
        }.get(metric)
    elif metric == "trainable_params":
        value = record.get("trainable_params")
    elif metric == "mean_step_time":
        value = record.get("mean_step_time")
    elif metric == "tokens_per_second":
        value = (
            record["tokens"] / (record["steps"] * record["mean_step_time"])
            if record.get("tokens")
            and record.get("steps")
            and record.get("mean_step_time")
            else None
        )
    elif metric == "inference_latency":
        value = deployed.get("ops", {}).get("inference_latency_mean_s")
    elif metric == "peak_memory":
        value = deployed.get("ops", {}).get("peak_memory_bytes")
    else:
        value = None
    return _numeric_values(value)


def metric_values_for_record(
    record: dict[str, Any], metric: str, rank: int
) -> list[float]:
    """Public read-only metric extraction shared by reports and visualizations."""
    return _report_metric_value(record, metric, rank)


def select_a3_k(
    records: list[dict[str, Any]],
    *,
    protocol: str,
    rank: int,
    order: tuple[str, ...],
    heldout: tuple[str, ...],
) -> dict[str, Any]:
    """Apply the report's validation-loss A3 K-selection rule."""
    candidates: dict[int, list[float]] = {}
    for row in records:
        if (
            row.get("method") == "a3"
            and row.get("protocol") == protocol
            and row.get("rank", 16) == rank
            and tuple(row.get("task_order", ())) == order
            and tuple(row.get("heldout_tasks", ())) == heldout
            and row.get("k") is not None
            and row.get("final_mean_loss") is not None
        ):
            candidates.setdefault(int(row["k"]), []).append(
                float(row["final_mean_loss"])
            )
    means = {k: float(np.mean(values)) for k, values in candidates.items()}
    selected = min(means, key=lambda value: (means[value], value)) if means else None
    return {
        "selected_k": selected,
        "mean_validation_loss_by_k": means,
        "rule": "lowest mean validation loss; smaller K breaks ties",
    }


def _format_config_label(method: str, rank: int | None, k: int | None) -> str:
    m = method.upper()
    parts: list[str] = []
    if rank is not None:
        parts.append(f"r={rank}")
    if k is not None:
        parts.append(f"K={k}")
    if parts:
        return f"{m} ({', '.join(parts)})"
    return m


def _append_metric_table(
    lines: list[str],
    title: str,
    metric_specs: list[tuple[str, str]],
    configs: list[tuple[str, int | None, int | None]],
    by_config: dict[tuple[str, int | None, int | None], list[dict[str, Any]]],
    report_rank: int,
) -> None:
    labels = [_format_config_label(m, r, k) for m, r, k in configs]
    lines.extend(
        [
            "",
            f"## {title}",
            "",
            f"| Metric | {' | '.join(labels)} | Coverage |",
            f"|---|{'---:|' * len(labels)}---|",
        ]
    )
    for label, key in metric_specs:
        cells: list[str] = []
        measured = False
        for cfg in configs:
            values = [
                value
                for record in by_config.get(cfg, [])
                for value in _report_metric_value(record, key, report_rank)
            ]
            if values:
                measured = True
                cells.append(f"{sum(values) / len(values):.6g}")
            else:
                cells.append("N/A")
        coverage = "measured" if measured else "not measured"
        lines.append(f"| {label} | {' | '.join(cells)} | {coverage} |")


def _cell(values: list[float]) -> str:
    if not values:
        return "N/A"
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else None
    return (
        f"{mean:.6g}"
        + (f" +/- {std:.3g}" if std is not None else "")
        + f" (n={len(values)})"
    )


def report_manifest(output_dir: Path, report_rank: int = 4) -> Path:
    evaluation = _read_json(output_dir / "evaluation.json")
    records = [row for row in evaluation["records"] if row.get("status") != "invalid"]
    invalid = [row for row in evaluation["records"] if row.get("status") == "invalid"]
    external_specs = EXTERNAL_METRIC_SPECS
    internal_specs = INTERNAL_METRIC_SPECS
    operational_specs = OPERATIONAL_METRIC_SPECS
    base_conditions = sorted(
        {
            (
                str(row.get("protocol", "sequential")),
                int(row.get("rank", 16)),
                tuple(row.get("task_order", EXPECTED_DOMAINS)),
                tuple(row.get("heldout_tasks", ())),
            )
            for row in records
        },
        key=lambda item: (item[0], item[1], item[2], item[3]),
    )
    lines = [
        "# A1/A2/A3 Continuous Residual Benchmark Report",
        "",
        f"- Completed valid runs: {len(records)}/{evaluation['planned_runs']}",
        f"- Invalid runs excluded: {len(invalid)}",
        f"- Domains: {', '.join(evaluation.get('domains', []))}",
        "- Every numeric comparison below uses seeds present for all three stages.",
        "",
        "## Run Completeness",
        "",
        "| Run | Method | Protocol | Rank | K | Status | Reason |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for row in evaluation["records"]:
        lines.append(
            f"| {row.get('run_id')} | {row.get('method')} | {row.get('protocol', 'N/A')} | "
            f"{row.get('rank', 'N/A')} | {row.get('k', 'N/A')} | "
            f"{row.get('status', 'valid')} | {row.get('reason', '')} |"
        )

    lines.extend(
        [
            "",
            "## A3 K Selection",
            "",
            "| Protocol | Rank | Order | Selected K | Validation loss | Rule |",
            "|---|---:|---|---:|---:|---|",
        ]
    )
    selected_k_by_condition: dict[tuple[Any, ...], int] = {}
    for protocol, rank, order, heldout in base_conditions:
        selection = select_a3_k(
            records,
            protocol=protocol,
            rank=rank,
            order=order,
            heldout=heldout,
        )
        selected_k = selection["selected_k"]
        if selected_k is not None:
            selected_k_by_condition[(protocol, rank, order, heldout)] = selected_k
            lines.append(
                f"| {protocol} | {rank} | {' -> '.join(order)} | {selected_k} | "
                f"{selection['mean_validation_loss_by_k'][selected_k]:.6g} | "
                f"{selection['rule']} |"
            )

    summary_candidates = []
    for protocol, rank, order, heldout in base_conditions:
        a3_k_values = sorted(
            {
                row.get("k")
                for row in records
                if row.get("method") == "a3"
                and row.get("protocol") == protocol
                and row.get("rank", 16) == rank
                and tuple(row.get("task_order", ())) == order
                and tuple(row.get("heldout_tasks", ())) == heldout
            },
            key=lambda value: -1 if value is None else int(value),
        ) or [None]
        for k in a3_k_values:
            method_rows = {}
            for method in ("a1", "a2", "a3"):
                method_rows[method] = [
                    row
                    for row in records
                    if row.get("method") == method
                    and row.get("protocol") == protocol
                    and row.get("rank", 16) == rank
                    and tuple(row.get("task_order", ())) == order
                    and tuple(row.get("heldout_tasks", ())) == heldout
                    and (method != "a3" or row.get("k") == k)
                ]
            title = (
                f"protocol={protocol}, rank={rank}, A3 K={k}, "
                f"order={' -> '.join(order)}"
            )
            if heldout:
                title += f", heldout={','.join(heldout)}"
            candidate_seeds = (
                set.intersection(
                    *[
                        {int(row["seed"]) for row in method_rows[method]}
                        for method in ("a1", "a2", "a3")
                    ]
                )
                if all(method_rows.values())
                else set()
            )
            fair_seeds = {
                seed
                for seed in candidate_seeds
                if len(
                    {
                        row.get("runtime_data_sha256")
                        for method in ("a1", "a2", "a3")
                        for row in method_rows[method]
                        if int(row["seed"]) == seed
                    }
                )
                == 1
            }
            lines.extend(["", f"## Comparison: {title}", ""])
            for table_title, specs in (
                ("External Metrics", external_specs),
                ("Internal Metrics", internal_specs),
                ("Operational Metrics", operational_specs),
            ):
                lines.extend(
                    [
                        f"### {table_title}",
                        "",
                        "| Metric | A1 | A2 | A3 | A2-A1 paired delta | A3-A1 paired delta | Direction | Matched seeds |",
                        "|---|---:|---:|---:|---:|---:|---|---:|",
                    ]
                )
                for label, metric, direction in specs:
                    by_method_seed: dict[str, dict[int, float]] = {}
                    for method in ("a1", "a2", "a3"):
                        seed_values = {}
                        for row in method_rows[method]:
                            values = _report_metric_value(row, metric, rank)
                            if values:
                                seed_values[int(row["seed"])] = float(np.mean(values))
                        by_method_seed[method] = seed_values
                    metric_seed_sets = [
                        set(by_method_seed[method]) for method in ("a1", "a2", "a3")
                    ]
                    metric_seeds = (
                        set.intersection(*metric_seed_sets)
                        if all(metric_seed_sets)
                        else set()
                    )
                    metric_seeds &= fair_seeds
                    cells = []
                    means = {}
                    for method in ("a1", "a2", "a3"):
                        values = [
                            by_method_seed[method][seed]
                            for seed in sorted(metric_seeds)
                        ]
                        cells.append(_cell(values))
                        means[method] = float(np.mean(values)) if values else None
                    a2_delta = [
                        by_method_seed["a2"][seed] - by_method_seed["a1"][seed]
                        for seed in sorted(metric_seeds)
                    ]
                    a3_delta = [
                        by_method_seed["a3"][seed] - by_method_seed["a1"][seed]
                        for seed in sorted(metric_seeds)
                    ]
                    lines.append(
                        f"| {label} | {' | '.join(cells)} | {_cell(a2_delta)} | "
                        f"{_cell(a3_delta)} | {direction} | {len(metric_seeds)} |"
                    )
                    selected_k = selected_k_by_condition.get(
                        (protocol, rank, order, heldout)
                    )
                    if (
                        metric in {"test_loss", "token_accuracy", "forgetting"}
                        and metric_seeds
                        and (k == selected_k or selected_k is None)
                    ):
                        summary_candidates.append((title, label, direction, means))

            matrix_seeds = {
                seed
                for seed in fair_seeds
                if all(
                    any(
                        int(row["seed"]) == seed and row.get("loss_matrix")
                        for row in method_rows[method]
                    )
                    for method in ("a1", "a2", "a3")
                )
            }
            if matrix_seeds:
                domains = next(
                    row.get("domains", [])
                    for row in method_rows["a1"]
                    if row.get("domains")
                )
                lines.extend(
                    [
                        "",
                        "### Mean Validation Loss Matrix",
                        "",
                        f"| Stage / checkpoint | {' | '.join(domains)} |",
                        f"|---|{'---:|' * len(domains)}",
                    ]
                )
                for method in ("a1", "a2", "a3"):
                    matrices = [
                        np.asarray(row["loss_matrix"], dtype=float)
                        for row in method_rows[method]
                        if int(row["seed"]) in matrix_seeds
                    ]
                    mean_matrix = np.mean(np.stack(matrices), axis=0)
                    for checkpoint, values in enumerate(mean_matrix):
                        cells = " | ".join(f"{value:.6g}" for value in values)
                        lines.append(
                            f"| {method.upper()} after checkpoint {checkpoint} | {cells} |"
                        )

    lines.extend(["", "## General Summary", ""])
    if not summary_candidates:
        lines.append(
            "No complete matched A1/A2/A3 cohort is available; no comparative conclusion is valid."
        )
    else:
        for title, label, direction, means in summary_candidates:
            measured = {
                method: value for method, value in means.items() if value is not None
            }
            if not measured:
                continue
            best = (
                min(measured, key=measured.get)
                if direction == "lower"
                else max(measured, key=measured.get)
            )
            lines.append(
                f"- {title}: {best.upper()} has the best measured {label.lower()} "
                f"({measured[best]:.6g}) among matched seeds."
            )
        measured_internal = sum(
            bool(_report_metric_value(row, metric, int(row.get("rank", report_rank))))
            for row in records
            for _, metric, _ in internal_specs
        )
        total_internal = len(records) * len(internal_specs)
        lines.append(
            f"- Internal metric coverage: {measured_internal}/{total_internal} run-metric cells."
        )
        lines.append(
            "- N/A means a prerequisite was not configured or no fully matched cohort exists; "
            "it is never interpreted as zero."
        )
        lines.append(
            "- This summary is descriptive and intentionally does not collapse heterogeneous "
            "metrics into a single composite score."
        )

    lines.extend(
        [
            "",
            "## Scope",
            "",
            "Symbolic metrics are intentionally excluded until A4 supplies checkpoint, bank, "
            "alignment labels, retrieval probes, and measurement artifacts. See manifest "
            "`a4_runs` for blocked A4 child specs.",
        ]
    )

    path = output_dir / "REPORT.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
