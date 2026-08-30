"""Unit tests for PyTorch -> JAX tensor conversion (substrate.conversion)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from substrate.conversion import ConversionReport, convert_state_dict


class TestConvertStateDict:
    def test_tied_weights_deduplicated(self):
        # Tied weights share memory and view
        w = torch.randn(10, 10, dtype=torch.float32)
        state_dict = {
            "transformer.wte.weight": w,
            "lm_head.weight": w,  # Tied to wte
        }
        result, report = convert_state_dict(state_dict, dedupe=True)
        assert len(result) == 2
        assert result["transformer.wte.weight"] is result["lm_head.weight"]
        assert report.num_tensors_converted == 2
        assert report.num_tensors_deduplicated == 1
        assert report.num_tensors_excluded == 0

    def test_sliced_weights_sharing_storage_not_deduplicated(self):
        # Two distinct slices of the same storage buffer must NOT be deduplicated
        buf = torch.randn(20, 10, dtype=torch.float32)
        s1 = buf[:10]
        s2 = buf[10:]
        state_dict = {"layer.q": s1, "layer.k": s2}
        result, report = convert_state_dict(state_dict, dedupe=True)
        assert len(result) == 2
        assert result["layer.q"] is not result["layer.k"]
        assert report.num_tensors_deduplicated == 0
        assert np.allclose(np.array(result["layer.q"]), s1.numpy())
        assert np.allclose(np.array(result["layer.k"]), s2.numpy())

    def test_non_floating_point_excluded_and_first_key(self):
        # Non-floating keys should be excluded without breaking subsequent keys
        int_tensor = torch.arange(10, dtype=torch.int64)
        float_tensor = torch.randn(10, dtype=torch.float32)
        state_dict = {
            "transformer.position_ids": int_tensor,
            "transformer.wte.weight": float_tensor,
        }
        result, report = convert_state_dict(state_dict)
        assert "transformer.position_ids" not in result
        assert "transformer.wte.weight" in result
        assert report.num_tensors_converted == 1
        assert report.num_tensors_excluded == 1
        assert report.excluded_keys == ("transformer.position_ids",)

    def test_bfloat16_conversion(self):
        bf_tensor = torch.randn(4, 4, dtype=torch.bfloat16)
        state_dict = {"weight": bf_tensor}

        # Default float32 conversion
        res_f32, rep_f32 = convert_state_dict(state_dict, dtype=jnp.float32)
        assert res_f32["weight"].dtype == jnp.float32
        assert rep_f32.dtype_policy == "float32"

        # Preserve dtype (bfloat16)
        res_raw, rep_raw = convert_state_dict(state_dict, dtype=None)
        assert res_raw["weight"].dtype == jnp.bfloat16
        assert rep_raw.dtype_policy == "preserve"

    def test_multithreaded_conversion(self):
        state_dict = {f"w_{i}": torch.randn(10, 10) for i in range(8)}
        result, report = convert_state_dict(state_dict, max_workers=4)
        assert len(result) == 8
        assert report.num_tensors_converted == 8

    def test_numpy_and_jax_inputs(self):
        np_arr = np.ones((3, 3), dtype=np.float32)
        jax_arr = jnp.zeros((3, 3), dtype=jnp.float32)
        state_dict = {"np_key": np_arr, "jax_key": jax_arr}
        result, report = convert_state_dict(state_dict)
        assert len(result) == 2
        assert report.num_tensors_converted == 2

    def test_unsupported_type_raises_typeerror(self):
        with pytest.raises(TypeError, match="Unsupported value type"):
            convert_state_dict({"invalid": "string_value"})

    def test_conversion_report_as_dict(self):
        state_dict = {
            "w1": torch.randn(5, 5),
            "mask": torch.ones(5, dtype=torch.bool),
        }
        _, report = convert_state_dict(state_dict)
        d = report.as_dict()
        assert d["num_tensors_converted"] == 1
        assert d["num_tensors_excluded"] == 1
        assert d["excluded_keys"] == ["mask"]
        assert "elapsed_seconds" in d
        assert report.bytes_saved_by_dedup >= 0