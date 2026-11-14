"""Language-model, continual-learning, and compute metrics."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypeVar, cast

import jax
import jax.numpy as jnp
import numpy as np

T = TypeVar("T")


def cross_entropy_loss(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Causal cross-entropy loss with next-token shifting and padding masking."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]

    log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
    mask = shift_labels != -100
    safe_labels = jnp.where(mask, shift_labels, 0)

    gathered_log_probs = jnp.take_along_axis(
        log_probs, safe_labels[..., None], axis=-1
    ).squeeze(-1)

    return -jnp.sum(gathered_log_probs * mask) / jnp.maximum(1.0, jnp.sum(mask))


def kl_to_base(base_logits: jax.Array, adapted_logits: jax.Array) -> jax.Array:
    """Mean token-level KL divergence from base to adapted distributions."""
    log_p = jax.nn.log_softmax(base_logits.astype(jnp.float32), axis=-1)
    log_q = jax.nn.log_softmax(adapted_logits.astype(jnp.float32), axis=-1)
    return jnp.mean(jnp.sum(jnp.exp(log_p) * (log_p - log_q), axis=-1))


def perplexity(loss: jax.Array | float) -> jax.Array:
    return jnp.exp(jnp.asarray(loss, dtype=jnp.float32))


def token_accuracy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Causal next-token top-1 accuracy, ignoring ``-100`` labels."""
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("expected logits [B,T,V] and labels [B,T]")
    target = labels[:, 1:]
    mask = target != -100
    predicted = jnp.argmax(logits[:, :-1], axis=-1)
    correct = (predicted == jnp.where(mask, target, 0)) & mask
    return jnp.sum(correct) / jnp.maximum(jnp.sum(mask), 1)


def calibration_stats(
    logits: jax.Array, labels: jax.Array, bins: int = 15
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return per-bin confidence, accuracy, and count sums for LM tokens."""
    if bins < 1:
        raise ValueError("bins must be >= 1")
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("expected logits [B,T,V] and labels [B,T]")
    target = labels[:, 1:]
    mask = (target != -100).astype(jnp.float32)
    probs = jax.nn.softmax(logits[:, :-1].astype(jnp.float32), axis=-1)
    confidence = jnp.max(probs, axis=-1)
    predicted = jnp.argmax(probs, axis=-1)
    safe_target = jnp.where(target != -100, target, 0)
    correct = (predicted == safe_target).astype(jnp.float32)

    indices = jnp.clip((confidence * bins).astype(jnp.int32), 0, bins - 1)
    confidence_sum = jax.ops.segment_sum(
        (confidence * mask).ravel(), indices.ravel(), num_segments=bins
    )
    accuracy_sum = jax.ops.segment_sum(
        (correct * mask).ravel(), indices.ravel(), num_segments=bins
    )
    counts = jax.ops.segment_sum(mask.ravel(), indices.ravel(), num_segments=bins)
    return confidence_sum, accuracy_sum, counts


def ece_from_stats(
    confidence_sum: jax.Array, accuracy_sum: jax.Array, counts: jax.Array
) -> jax.Array:
    """Compute token ECE from pooled per-bin sufficient statistics."""
    total = jnp.maximum(counts.sum(), 1.0)
    return jnp.abs(confidence_sum - accuracy_sum).sum() / total


def expected_calibration_error(
    logits: jax.Array, labels: jax.Array, bins: int = 15
) -> jax.Array:
    """Token-level ECE using confidence of the top-1 next-token prediction."""
    confidence_sum, accuracy_sum, counts = calibration_stats(logits, labels, bins=bins)
    return ece_from_stats(confidence_sum, accuracy_sum, counts)


def paired_deltas(
    values: dict[str, dict[int, float]], reference: str
) -> dict[str, dict[int, float]]:
    """Return method-minus-reference values paired by seed."""
    if reference not in values:
        raise KeyError(reference)
    return {
        method: {
            seed: float(seed_values[seed] - values[reference][seed])
            for seed in seed_values
            if seed in values[reference]
        }
        for method, seed_values in values.items()
        if method != reference
    }


def sample_mean_std(values: list[float]) -> tuple[float | None, float | None]:
    """Return sample mean and sample standard deviation."""
    if not values:
        return None, None
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else None
    return mean, std


def _loss_matrix(values: Any, t: int) -> jax.Array:
    loss = jnp.asarray(values, dtype=jnp.float32)
    if loss.ndim != 2:
        raise ValueError(f"Expected a 2-D loss matrix L[t, j], got shape {loss.shape}.")
    if not isinstance(t, int) or t < 0:
        raise ValueError(f"t must be an integer >= 0, got {t!r}.")
    if t >= min(loss.shape):
        raise ValueError(f"t={t} is out of range for L with shape {loss.shape}.")
    return loss


def mean_seen_loss(values: Any, t: int) -> jax.Array:
    """Return mean loss on tasks seen through training step ``t``."""
    loss = _loss_matrix(values, t)
    seen = loss[t, : t + 1]
    valid = jnp.isfinite(seen)
    count = jnp.sum(valid)
    return jnp.where(
        count > 0,
        jnp.sum(jnp.where(valid, seen, 0.0)) / jnp.maximum(count, 1),
        jnp.nan,
    )


def mean_forgetting(values: Any, t: int) -> jax.Array:
    """Return mean increase from each prior task's best observed loss."""
    loss = _loss_matrix(values, t)
    if t == 0:
        raise ValueError("Forgetting is undefined at t=0: there are no earlier tasks.")
    training_step = jnp.arange(t)[:, None]
    task = jnp.arange(t)[None, :]
    valid_history = (training_step >= task) & jnp.isfinite(loss[:t, :t])
    best = jnp.min(jnp.where(valid_history, loss[:t, :t], jnp.inf), axis=0)
    increase = loss[t, :t] - best
    valid_increase = jnp.isfinite(best) & jnp.isfinite(loss[t, :t])
    count = jnp.sum(valid_increase)
    return jnp.where(
        count > 0,
        jnp.sum(jnp.where(valid_increase, increase, 0.0)) / jnp.maximum(count, 1),
        jnp.nan,
    )


def training_step_time(
    step_fn: Callable[..., T], *args: Any, **kwargs: Any
) -> tuple[T, float]:
    start = time.perf_counter()
    out = step_fn(*args, **kwargs)
    jax.block_until_ready(out)
    return out, time.perf_counter() - start


# =============================================================================
# External-metric additions: label-token probe scores, continual-learning transfer,
# sequential-vs-joint gap, anchor drift and the long-range proxy.
# =============================================================================
def classification_scores(
    logits: jax.Array, positions: jax.Array, label_token_ids: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Predicted class and its confidence for a label-token probe.

    Same class scores as ``classification_predictions`` (best token per class), turned
    into a distribution over the *classes* with a softmax, so the confidence is
    comparable to accuracy for task-level calibration.
    """
    if logits.ndim != 3 or positions.ndim != 1 or label_token_ids.ndim != 2:
        raise ValueError("invalid logits, positions, or label-token shapes")
    if logits.shape[0] != positions.shape[0]:
        raise ValueError("one position is required for each example")
    scores = logits[jnp.arange(logits.shape[0]), positions][:, label_token_ids]
    class_scores = jnp.max(scores, axis=-1).astype(jnp.float32)
    probs = jax.nn.softmax(class_scores, axis=-1)
    return jnp.argmax(probs, axis=-1), jnp.max(probs, axis=-1)


def calibration_from_predictions(
    confidences: Any, correct: Any, bins: int = 15
) -> dict[str, float | int | None]:
    """ECE (and mean confidence/accuracy) from per-example confidence and 0/1 correctness."""
    conf = np.asarray(confidences, dtype=float)
    hit = np.asarray(correct, dtype=float)
    if conf.shape != hit.shape or conf.ndim != 1:
        raise ValueError("confidences and correct must be 1-D arrays of equal length")
    if bins < 1:
        raise ValueError("bins must be >= 1")
    if conf.size == 0:
        return {"ece": None, "mean_confidence": None, "accuracy": None, "n": 0}
    idx = np.clip((conf * bins).astype(int), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        sel = idx == b
        if sel.any():
            ece += abs(conf[sel].sum() - hit[sel].sum())
    return {
        "ece": float(ece / conf.size),
        "mean_confidence": float(conf.mean()),
        "accuracy": float(hit.mean()),
        "n": int(conf.size),
    }


def multilabel_hit_metrics(
    gold_sets: Any, predictions: Any, n_classes: int
) -> dict[str, float]:
    """Metrics for multi-label tasks (EUR-Lex) when the model emits one label per example.

    ``hit_at_1``: the predicted label is one of the gold labels. ``micro_*`` treat each
    (example, label) pair: the single prediction is a true positive if gold, otherwise a
    false positive, and every unpredicted gold label is a false negative. Single-label
    probes cannot recover all gold labels, so recall is bounded by 1 / mean(|gold|); read
    ``hit_at_1`` and ``micro_precision`` as the primary numbers.
    """
    golds = [{int(x) for x in g} for g in gold_sets]
    preds = [int(x) for x in np.asarray(predictions).ravel()]
    if len(golds) != len(preds) or not golds:
        raise ValueError("need one prediction per (non-empty) gold set")
    if any(not 0 <= p < n_classes for p in preds):
        raise ValueError("prediction outside [0, n_classes)")
    tp = sum(1 for g, p in zip(golds, preds, strict=True) if p in g)
    fp = len(preds) - tp
    fn = sum(len(g) - (1 if p in g else 0) for g, p in zip(golds, preds, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "hit_at_1": tp / len(golds),
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ),
    }


# =============================================================================
# Continual-learning matrices: backward / held-out transfer, sequential vs joint
# =============================================================================
def backward_transfer(matrix: Any, t: int, *, higher_is_better: bool = True) -> float:
    """Backward transfer after task ``t``: mean over ``j < t`` of ``A[t, j] - A[j, j]``.

    Compares each earlier task now with the moment it was learned. Sign-adjusted so
    **negative = forgetting, positive = later tasks helped earlier ones**, for both
    accuracy-like (higher better) and loss-like (lower better) metrics. Unlike
    ``mean_forgetting`` it does not use each task's peak, only its value right after
    it was trained. ``nan`` if no earlier task has finite values.
    """
    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2 or not isinstance(t, int) or t < 1 or t >= min(a.shape):
        raise ValueError(f"t must be in [1, {min(a.shape) - 1}] for matrix {a.shape}")
    sign = 1.0 if higher_is_better else -1.0
    diffs = [
        sign * (a[t, j] - a[j, j])
        for j in range(t)
        if np.isfinite(a[t, j]) and np.isfinite(a[j, j])
    ]
    return float(np.mean(diffs)) if diffs else float("nan")


def heldout_transfer(
    loss: Any,
    base_loss: Any,
    names: list[str] | None = None,
    min_in_domain: float = 0.02,
) -> dict[str, Any]:
    """Zero-shot transfer to domains the adapter has not been trained on yet.

    ``loss[t][j]``: loss on domain ``j`` after training tasks ``0..t``; ``base_loss[j]``:
    the frozen model's loss. For ``j > t`` the domain is held out, and
    ``gain[t][j] = base_loss[j] - loss[t][j]`` (positive = training on the earlier
    domains helps the unseen one). ``ratio[t][j]`` divides that by the in-domain gain
    ``base_loss[j] - loss[j][j]`` (``None`` unless the in-domain gain exceeds
    ``min_in_domain``). ``final`` is the standard protocol: train on all but the last
    domain, test on the last.
    """
    a = np.asarray(loss, dtype=float)
    base = np.asarray(base_loss, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or base.shape != (a.shape[1],):
        raise ValueError(
            f"loss must be square [T, T] and base_loss [T]; got {a.shape}, {base.shape}"
        )
    n = a.shape[0]
    names = names or [str(i) for i in range(n)]
    gain: list[list[float | None]] = [[None] * n for _ in range(n)]
    ratio: list[list[float | None]] = [[None] * n for _ in range(n)]
    per_checkpoint: list[float | None] = []
    for t in range(n):
        vals = []
        for j in range(t + 1, n):
            g = float(base[j] - a[t, j])
            if np.isfinite(g):
                gain[t][j] = g
                vals.append(g)
                own = float(base[j] - a[j, j])
                if np.isfinite(own) and own >= min_in_domain:
                    ratio[t][j] = g / own
        per_checkpoint.append(float(np.mean(vals)) if vals else None)
    final = None
    if n >= 2 and gain[n - 2][n - 1] is not None:
        final = {
            "train_on": names[: n - 1],
            "test_on": names[n - 1],
            "gain": gain[n - 2][n - 1],
            "ratio": ratio[n - 2][n - 1],
        }
    return {
        "gain": gain,
        "ratio": ratio,
        "per_checkpoint": per_checkpoint,
        "final": final,
    }


def sequential_joint_gap(
    sequential: Any,
    joint: Any,
    base: Any | None = None,
    *,
    higher_is_better: bool = False,
    min_improvement: float = 0.02,
) -> dict[str, Any]:
    """Cost of learning sequentially instead of jointly, per domain and on average.

    ``sequential`` / ``joint``: final per-domain values of the same metric after the
    two protocols (matched training budget). ``gap`` is sign-adjusted so **positive =
    sequential is worse**: ``seq - joint`` for loss, ``joint - seq`` for accuracy.
    ``relative_gap`` divides by the joint model's improvement over ``base`` (guarded:
    ``None`` unless that improvement is at least ``min_improvement``).
    """
    s = np.asarray(sequential, dtype=float)
    j = np.asarray(joint, dtype=float)
    if s.shape != j.shape or s.ndim != 1 or s.size == 0:
        raise ValueError(
            "sequential and joint must be equal-length non-empty 1-D arrays"
        )
    gap = (j - s) if higher_is_better else (s - j)
    out: dict[str, Any] = {
        "gap": gap.tolist(),
        "mean_gap": float(np.nanmean(gap)) if np.isfinite(gap).any() else None,
        "relative_gap": None,
    }
    if base is not None:
        b = np.asarray(base, dtype=float)
        if b.shape != s.shape:
            raise ValueError("base must match the shape of sequential")
        gain = (j - b) if higher_is_better else (b - j)
        rel = [
            (
                float(g / d)
                if np.isfinite(g) and np.isfinite(d) and d >= min_improvement
                else None
            )
            for g, d in zip(gap, gain, strict=True)
        ]
        out["relative_gap"] = rel
        usable = [r for r in rel if r is not None]
        out["mean_relative_gap"] = float(np.mean(usable)) if usable else None
    return out


# =============================================================================
# Anchor drift and long-range behaviour
# =============================================================================
def anchor_drift(
    kl_by_checkpoint: Any, loss_by_checkpoint: Any, base_loss: float
) -> dict[str, Any]:
    """Drift from the frozen model on a fixed anchor corpus, at every checkpoint.

    ``kl_by_checkpoint[t]``: KL(base || adapted) after task ``t``; ``loss_by_checkpoint[t]``:
    adapted loss on the same corpus. Reports the final and peak KL and the loss change
    relative to base (positive = the adapter made the anchor text worse).
    """
    kl = np.asarray(kl_by_checkpoint, dtype=float)
    loss = np.asarray(loss_by_checkpoint, dtype=float)
    if kl.shape != loss.shape or kl.ndim != 1 or kl.size == 0:
        raise ValueError("kl and loss must be equal-length non-empty 1-D arrays")
    return {
        "kl": kl.tolist(),
        "loss_change": (loss - float(base_loss)).tolist(),
        "final_kl": float(kl[-1]),
        "peak_kl": float(np.nanmax(kl)),
        "final_loss_change": float(loss[-1] - base_loss),
    }


def position_loss_profile(
    logits: jax.Array, labels: jax.Array, n_buckets: int = 4
) -> np.ndarray:
    """Mean next-token loss per position bucket (early ... late) of a long sequence."""
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("expected logits [B,T,V] and labels [B,T]")
    if n_buckets < 2 or labels.shape[1] - 1 < n_buckets:
        raise ValueError("need n_buckets >= 2 and at least n_buckets positions")
    logp = jax.nn.log_softmax(logits[:, :-1].astype(jnp.float32), axis=-1)
    target = labels[:, 1:]
    mask = (target != -100).astype(jnp.float32)
    nll = -jnp.take_along_axis(
        logp, jnp.where(target != -100, target, 0)[..., None], axis=-1
    )[..., 0]
    per_pos = np.asarray(
        jnp.sum(nll * mask, axis=0) / jnp.maximum(jnp.sum(mask, axis=0), 1.0)
    )
    return cast(
        np.ndarray,
        np.asarray([c.mean() for c in np.array_split(per_pos, n_buckets)], dtype=float),
    )


def long_range_summary(adapted_profile: Any, base_profile: Any) -> dict[str, Any]:
    """Long-range coherence from position-bucketed losses (adapted vs frozen model).

    ``context_gain`` = early-bucket loss minus late-bucket loss: how much more the model
    benefits from long context (larger = better use of context). ``delta_vs_base`` is the
    adapted-minus-base context gain, so a positive value means the adapter *improves*
    long-range use beyond what the frozen model already does.
    """
    a = np.asarray(adapted_profile, dtype=float)
    b = np.asarray(base_profile, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or a.size < 2:
        raise ValueError("profiles must be equal-length 1-D arrays with >= 2 buckets")
    gain_a, gain_b = float(a[0] - a[-1]), float(b[0] - b[-1])
    return {
        "adapted_profile": a.tolist(),
        "base_profile": b.tolist(),
        "context_gain": gain_a,
        "base_context_gain": gain_b,
        "delta_vs_base": gain_a - gain_b,
        "late_loss_change": float(a[-1] - b[-1]),
    }


def unseen_domain_transfer(
    final_loss: Any,
    base_loss: Any,
    trained: Any,
    unseen: Any,
    min_in_domain: float = 0.02,
) -> dict[str, Any]:
    """Transfer of the final adapter to domains it was never trained on (held-out protocol).

    ``final_loss[j]`` / ``base_loss[j]``: adapted and frozen-model loss on domain ``j``;
    ``trained`` / ``unseen``: index lists. ``gain`` = base - adapted per unseen domain
    (positive = the adapter helps without training on it); ``ratio`` divides by the mean
    in-domain gain, and is ``None`` unless that gain is at least ``min_in_domain``.
    """
    f = np.asarray(final_loss, dtype=float)
    b = np.asarray(base_loss, dtype=float)
    if f.shape != b.shape or f.ndim != 1:
        raise ValueError("final_loss and base_loss must be equal-length 1-D arrays")
    trained_idx, unseen_idx = [int(i) for i in trained], [int(i) for i in unseen]
    if not trained_idx or not unseen_idx:
        raise ValueError("need at least one trained and one unseen domain")
    in_domain = float(np.nanmean(b[trained_idx] - f[trained_idx]))
    gains = {j: float(b[j] - f[j]) for j in unseen_idx}
    guarded = np.isfinite(in_domain) and in_domain >= min_in_domain
    return {
        "in_domain_gain": in_domain,
        "gain": gains,
        "ratio": {j: (g / in_domain if guarded else None) for j, g in gains.items()},
        "mean_gain": float(np.mean(list(gains.values()))),
        "mean_ratio": (
            float(np.mean([g / in_domain for g in gains.values()])) if guarded else None
        ),
    }


def classification_metrics(
    targets: Any, predictions: Any, *, labels: Any | None = None
) -> dict[str, float]:
    target_array = np.asarray(targets)
    prediction_array = np.asarray(predictions)
    if target_array.shape != prediction_array.shape:
        raise ValueError(
            f"targets and predictions must have equal shapes, got "
            f"{target_array.shape} and {prediction_array.shape}"
        )
    if target_array.ndim != 1 or target_array.size == 0:
        raise ValueError("targets and predictions must be non-empty 1-D arrays")
    classes = (
        list(labels)
        if labels is not None
        else sorted(
            set(target_array.tolist()) | set(prediction_array.tolist()), key=str
        )
    )
    if not classes:
        raise ValueError("labels must contain at least one class")
    true_positive = []
    false_positive = []
    false_negative = []
    for label in classes:
        target_mask = target_array == label
        prediction_mask = prediction_array == label
        true_positive.append(int(np.count_nonzero(target_mask & prediction_mask)))
        false_positive.append(int(np.count_nonzero(~target_mask & prediction_mask)))
        false_negative.append(int(np.count_nonzero(target_mask & ~prediction_mask)))
    precision = [
        tp / (tp + fp) if tp + fp else 0.0
        for tp, fp in zip(true_positive, false_positive, strict=True)
    ]
    recall = [
        tp / (tp + fn) if tp + fn else 0.0
        for tp, fn in zip(true_positive, false_negative, strict=True)
    ]
    f1 = [
        2 * p * r / (p + r) if p + r else 0.0
        for p, r in zip(precision, recall, strict=True)
    ]
    return {
        "accuracy": float(np.mean(target_array == prediction_array)),
        "macro_f1": float(np.mean(f1)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
    }


def accuracy_forgetting(accuracy_matrix: Any, t: int) -> jax.Array:
    """Mean drop of each earlier task's accuracy from its best earlier value.

    ``accuracy_matrix[s, j]`` is accuracy on domain ``j`` after training step ``s``;
    non-finite entries are ignored. Undefined at ``t=0``.
    """
    a = jnp.asarray(accuracy_matrix, dtype=jnp.float32)
    if a.ndim != 2 or t < 0 or t >= min(a.shape):
        raise ValueError(f"t={t} is out of range for matrix of shape {a.shape}")
    if t == 0:
        raise ValueError("Forgetting is undefined at t=0: there are no earlier tasks.")
    step = jnp.arange(t)[:, None]
    task = jnp.arange(t)[None, :]
    valid_history = (step >= task) & jnp.isfinite(a[:t, :t])
    peak = jnp.max(jnp.where(valid_history, a[:t, :t], -jnp.inf), axis=0)
    drop = peak - a[t, :t]
    valid = jnp.isfinite(peak) & jnp.isfinite(a[t, :t])
    count = jnp.sum(valid)
    return jnp.where(
        count > 0,
        jnp.sum(jnp.where(valid, drop, 0.0)) / jnp.maximum(count, 1),
        jnp.nan,
    )


__all__ = [
    "accuracy_forgetting",
    "classification_metrics",
    "calibration_stats",
    "cross_entropy_loss",
    "ece_from_stats",
    "expected_calibration_error",
    "kl_to_base",
    "mean_forgetting",
    "mean_seen_loss",
    "paired_deltas",
    "perplexity",
    "sample_mean_std",
    "token_accuracy",
    "training_step_time",
    "anchor_drift",
    "backward_transfer",
    "calibration_from_predictions",
    "classification_scores",
    "heldout_transfer",
    "long_range_summary",
    "multilabel_hit_metrics",
    "position_loss_profile",
    "sequential_joint_gap",
    "unseen_domain_transfer",
]
