#!/usr/bin/env python3
"""Regression contracts for the experimental gfx1151 direct-WMMA FA path."""

from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "ggml/src/ggml-cuda/fattn-wmma-gfx1151.cuh"


class TestGfx1151WmmaIntrinsicContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = SRC.read_text()

    def test_gqa8_maps_two_q_heads_to_each_shared_kv_workgroup(self):
        """Four waves/head bound accumulator pressure while two heads reuse K/V."""
        self.assertRegex(self.source, r"const int wave_id\s*=\s*threadIdx\.x\s*/\s*WARP_SIZE")
        self.assertRegex(self.source, r"head_slot\s*=\s*wave_id\s*/\s*4")
        self.assertRegex(self.source, r"pv_quarter\s*=\s*wave_id\s*%\s*4")
        self.assertRegex(self.source, r"head_q\s*=\s*head_kv\s*\*\s*gqa_ratio\s*\+\s*gqa_group\s*\*\s*2\s*\+\s*head_slot")
        self.assertIn("dim3 grid(ntiles_x, 4 * n_head_kv, n_seq)", self.source)
        self.assertRegex(self.source, r"dim3 block\(\s*8\s*\*\s*WARP_SIZE")
        self.assertNotIn("dim3 grid(ntiles_x, n_head_q, n_seq)", self.source)

    def test_pv_accumulator_is_split_across_four_waves(self):
        self.assertRegex(self.source, r"float8_amdgcn\s+o_acc\[4\]")
        self.assertNotRegex(self.source, r"float8_amdgcn\s+o_acc\[(8|16)\]")

    def test_gfx1151_wmma_accumulator_coordinates_match_hardware_probe(self):
        self.assertRegex(self.source, r"m_row\s*=\s*2\s*\*\s*i\s*\+\s*lane\s*/\s*16")
        self.assertRegex(self.source, r"n_cur\s*=\s*k_step0\s*\+\s*col_idx")
        self.assertRegex(self.source, r"__shfl_xor\(r_max,\s*8\)")
        self.assertRegex(self.source, r"row_(max|sum)\[8\]")

    def test_pv_output_matches_hardware_probed_fragment_rows(self):
        self.assertRegex(self.source, r"v_col\s*=\s*\(pv_quarter\s*\*\s*4\s*\+\s*s\)\s*\*\s*16\s*\+\s*col_idx")

    def test_only_one_column_publishes_shared_softmax_metadata(self):
        self.assertRegex(self.source, re.compile(r"if\s*\(col_idx\s*==\s*0\)\s*\{[^}]*O_scale_lds", re.S))

    def test_dispatch_rejects_non_gqa8_shapes(self):
        """The specialized eight-wave kernel must not silently handle other GQA ratios."""
        self.assertRegex(self.source, r"gqa_ratio\s*!=\s*8")

    def test_experimental_intrinsic_path_is_opt_in(self):
        self.assertIn("GGML_CUDA_EXPERIMENTAL_GFX1151_INTRINSIC_WMMA", self.source)

    def test_mask_scan_handles_non_256_multiple_contexts(self):
        """20,000-token KV must receive a valid pruning bound rather than nullptr."""
        self.assertIn("flash_attn_mask_to_KV_max_tail_safe<16>", self.source)
        self.assertNotRegex(self.source, r"K->ne\[1\]\s*%\s*FATTN_KQ_STRIDE\s*==\s*0")
        self.assertRegex(self.source, r"ceil_div\(\s*n_kv\s*,\s*FATTN_KQ_STRIDE\s*\)")
        self.assertRegex(self.source, r"q0\s*\+\s*j\s*<\s*n_q")
        self.assertRegex(self.source, r"key\s*\+\s*1\s*<\s*n_kv")

    def test_unsupported_attention_sinks_fall_back_before_allocations(self):
        sink_guard = self.source.index("if (sinks != nullptr)")
        first_alloc = self.source.index("K_f16.alloc")
        self.assertLess(sink_guard, first_alloc)

    def test_q4_path_is_not_claimed_as_fused(self):
        """Current implementation globally converts Q4 K/V; name this honestly in source."""
        self.assertIn("WMMA_GFX1151_GLOBAL_KV_DEQUANT", self.source)


if __name__ == "__main__":
    unittest.main()
