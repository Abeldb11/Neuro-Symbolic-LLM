"""Tests for the external-metric math appended to ``metrics/performance.py``."""

from __future__ import annotations

import numpy as np
import pytest

from metrics.performance import (
    accuracy_forgetting,
    anchor_drift,
    backward_transfer,
    heldout_transfer,
    sequential_joint_gap,
)


def test_accuracy_forgetting_is_drop_from_best_earlier_value() -> None:
    acc = [[0.90, 0.20, 0.10], [0.70, 0.80, 0.15], [0.60, 0.75, 0.85]]
    assert float(accuracy_forgetting(acc, 1)) == pytest.approx(0.20)
    assert float(accuracy_forgetting(acc, 2)) == pytest.approx((0.30 + 0.05) / 2)
    with pytest.raises(ValueError):
        accuracy_forgetting(acc, 0)


def test_classification_metrics_macro_f1() -> None:
    from metrics.performance import classification_metrics

    out = classification_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert out["accuracy"] == pytest.approx(0.75)
    assert out["macro_f1"] == pytest.approx(
        (2 * 1 * 0.5 / 1.5 + 2 * (2 / 3) * 1 / (5 / 3)) / 2
    )


def test_classification_scores_confidence_is_class_softmax() -> None:
    import jax.numpy as jnp

    from metrics.performance import classification_scores

    logits = jnp.zeros((2, 3, 5)).at[0, 1, 2].set(4.0).at[1, 2, 3].set(1.0)
    pred, conf = classification_scores(
        logits, jnp.array([1, 2]), jnp.array([[0], [2], [3]])
    )
    assert pred.tolist() == [1, 2]
    assert float(conf[0]) == pytest.approx(float(np.exp(4) / (np.exp(4) + 2)), rel=1e-5)
    assert float(conf[1]) == pytest.approx(float(np.exp(1) / (np.exp(1) + 2)), rel=1e-5)


def test_calibration_from_predictions() -> None:
    from metrics.performance import calibration_from_predictions

    perfect = calibration_from_predictions([1.0, 1.0, 0.0001], [1, 1, 0])
    assert perfect["ece"] == pytest.approx(0.0001 / 3, abs=1e-6)
    over = calibration_from_predictions([0.9] * 10, [1] * 5 + [0] * 5)
    assert over["ece"] == pytest.approx(0.4)
    assert calibration_from_predictions([], [])["ece"] is None
    with pytest.raises(ValueError):
        calibration_from_predictions([0.5], [1, 0])


def test_multilabel_hit_metrics() -> None:
    from metrics.performance import multilabel_hit_metrics

    out = multilabel_hit_metrics([{0, 1}, {2}, {1}], [1, 0, 1], n_classes=3)
    assert out["hit_at_1"] == pytest.approx(2 / 3)
    assert out["micro_precision"] == pytest.approx(2 / 3)
    assert out["micro_recall"] == pytest.approx(2 / 4)  # 4 gold labels, 2 recovered
    with pytest.raises(ValueError):
        multilabel_hit_metrics([{0}], [5], n_classes=3)


def test_backward_transfer_sign_convention() -> None:

    acc = [[0.8, 0.0, 0.0], [0.6, 0.9, 0.0], [0.5, 0.7, 0.9]]
    # j<2: (0.5-0.8) and (0.7-0.9) -> mean -0.25 (forgetting)
    assert backward_transfer(acc, 2) == pytest.approx(-0.25)
    loss = [[2.0, 3, 3], [2.5, 2.0, 3], [2.6, 2.1, 2.0]]
    assert backward_transfer(loss, 2, higher_is_better=False) == pytest.approx(
        -(0.6 + 0.1) / 2
    )
    with pytest.raises(ValueError):
        backward_transfer(acc, 0)
    assert np.isnan(backward_transfer([[np.nan, 0], [np.nan, 1]], 1))


def test_heldout_transfer() -> None:

    base = [3.0, 3.0, 3.0]
    loss = [[2.0, 2.8, 2.9], [2.1, 2.0, 2.5], [2.2, 2.1, 2.0]]
    out = heldout_transfer(loss, base, ["a", "b", "c"])
    assert out["gain"][0][1] == pytest.approx(0.2)
    assert out["gain"][1][2] == pytest.approx(0.5)
    assert out["gain"][1][0] is None  # seen domain, not held out
    assert out["per_checkpoint"][0] == pytest.approx((0.2 + 0.1) / 2)
    assert out["per_checkpoint"][2] is None
    assert out["ratio"][1][2] == pytest.approx(0.5 / 1.0)
    assert out["final"] == {
        "train_on": ["a", "b"],
        "test_on": "c",
        "gain": pytest.approx(0.5),
        "ratio": pytest.approx(0.5),
    }
    with pytest.raises(ValueError):
        heldout_transfer([[1.0, 2.0]], [1.0, 2.0])


def test_sequential_joint_gap_signs_and_guard() -> None:

    out = sequential_joint_gap([2.5, 2.2], [2.0, 2.2], base=[3.0, 2.21])
    assert out["gap"] == pytest.approx([0.5, 0.0])  # positive = sequential worse
    assert out["mean_gap"] == pytest.approx(0.25)
    assert out["relative_gap"][0] == pytest.approx(0.5)
    assert out["relative_gap"][1] is None  # joint barely beat base: ratio unreliable
    acc = sequential_joint_gap([0.6], [0.7], higher_is_better=True)
    assert acc["gap"] == pytest.approx([0.1])
    with pytest.raises(ValueError):
        sequential_joint_gap([1.0], [1.0, 2.0])


def test_anchor_drift() -> None:

    out = anchor_drift([0.0, 0.1, 0.05], [3.0, 3.2, 3.1], base_loss=3.0)
    assert out["final_kl"] == pytest.approx(0.05)
    assert out["peak_kl"] == pytest.approx(0.1)
    assert out["loss_change"] == pytest.approx([0.0, 0.2, 0.1])
    with pytest.raises(ValueError):
        anchor_drift([0.1], [1.0, 2.0], 1.0)


def test_position_profile_and_long_range_summary() -> None:
    import jax.numpy as jnp

    from metrics.performance import long_range_summary, position_loss_profile

    v, t = 5, 9
    labels = jnp.zeros((2, t), jnp.int32)
    # early positions confidently wrong, late positions confidently right
    logits = jnp.zeros((2, t, v)).at[:, :4, 1].set(8.0).at[:, 4:, 0].set(8.0)
    prof = position_loss_profile(logits, labels, n_buckets=2)
    assert prof[0] > 5.0 and prof[1] < 0.01
    summary = long_range_summary(prof, np.array([4.0, 4.0]))
    assert summary["context_gain"] == pytest.approx(prof[0] - prof[1])
    assert summary["delta_vs_base"] == pytest.approx(prof[0] - prof[1])
    with pytest.raises(ValueError):
        position_loss_profile(logits, labels, n_buckets=1)
