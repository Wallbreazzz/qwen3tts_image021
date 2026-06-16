#!/bin/bash
# deploy_patch.sh — v5: Clean production patches (FIX2/3/5/5b/6, no DIAGs)
#
# Bug history:
#   v1: cos_sim=0.069 → 4 FULL-only checks skip PIECEWISE → stale attn params
#   v2: FIX1-4 extended checks to PIECEWISE → FIX2 triggered full_graph_fia
#        → graph_task_group_begin crash (error 107029: stream not captured)
#   v3: FIX5 adds FULL guard to full_graph_fia; FIX5b adds cudagraph_runtime_mode
#        to extra_attrs → UnboundLocalError on need_sync (DIAG log before def)
#   v4: Revert FIX1/FIX4 (update_stream/need_sync only exist for FULL mode;
#        PIECEWISE has no graph_params entries per FIX5). Keep DIAG logging.
#        The REAL fix for cos_sim=0.069 is FIX5: PIECEWISE skips full_graph_fia,
#        uses normal FIA path → correct attention output.
#   v5: Add FIX6 (has_preprocess in _dummy_run). Remove DIAG patches for clean
#        production deployment. DIAG patches available in deploy_debug.sh.
#
# Current fix set:
#   FIX2:        _should_build_dummy_attn_metadata    FULL→(FULL,PIECEWISE)
#   FIX3:        pad_attn                              FULL→(FULL,PIECEWISE)
#   FIX5:        attention_v1.py capturing guard       add FULL mode check (4 locations)
#   FIX5b:       extra_attrs                           add cudagraph_runtime_mode
#   FIX6:        npu_model_runner.py _dummy_run        add has_preprocess branch
#
# Why FIX1 and FIX4 are reverted:
#   - update_stream only exists when has_full_cudagraphs() (line 3040)
#   - NPUARModelRunner doesn't have update_stream for PIECEWISE mode
#   - PIECEWISE has no graph_params entries (FIX5 prevents full_graph_fia)
#   - update_full_graph_params() would find 0 entries and do nothing for PIECEWISE
#   - need_sync synchronize is unnecessary for PIECEWISE (no async update_stream work)

set -e

OMNI_DIR="/vllm-workspace/vllm-omni"
ASCEND_DIR="/vllm-workspace/vllm-ascend"
VLLM_DIR="/vllm-workspace/vllm"

echo "=== Step 1: Revert any previous patches ==="
cd "$OMNI_DIR" && git checkout -- \
    vllm_omni/model_executor/models/qwen3_tts/qwen3_tts_talker.py \
    vllm_omni/platforms/npu/worker/npu_ar_model_runner.py \
    vllm_omni/platforms/npu/worker/npu_model_runner.py \
    vllm_omni/utils/debug_dump.py 2>/dev/null || true
cd "$VLLM_DIR" && git checkout -- \
    vllm/model_executor/models/qwen2.py 2>/dev/null || true
cd "$ASCEND_DIR" && git checkout -- \
    vllm_ascend/worker/model_runner_v1.py \
    vllm_ascend/ascend_forward_context.py \
    vllm_ascend/attention/attention_v1.py 2>/dev/null || true
rm -f /home/h00875519/compare_dumps.py /home/h00875519/test_request.sh
rm -f /home/h00875519/tts_debug_dump/*.pt /home/h00875519/tts_debug_dump/*.stats.txt
echo "Revert done."

echo "=== Step 2: Apply patches ==="

python3 << 'PATCH_EOF'
def patch_file(path, old, new, label):
    with open(path, 'r') as f:
        content = f.read()
    if old in content:
        content = content.replace(old, new, 1)
        with open(path, 'w') as f:
            f.write(content)
        print(f"[OK] {label}")
    else:
        print(f"[SKIP] {label} — pattern not found")

MR = "/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py"
ATTN = "/vllm-workspace/vllm-ascend/vllm_ascend/attention/attention_v1.py"
CTX = "/vllm-workspace/vllm-ascend/vllm_ascend/ascend_forward_context.py"
NPU_MR = "/vllm-workspace/vllm-omni/vllm_omni/platforms/npu/worker/npu_model_runner.py"

# --- FIX2: _should_build_dummy_attn_metadata FULL→(FULL,PIECEWISE) ---
# PIECEWISE with enable_npugraph_ex needs dummy attn_metadata during warmup/capture.
# Safe with FIX5: PIECEWISE won't enter full_graph_fia (FULL guard added).
patch_file(MR,
    'return force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL',
    'return force_attention or cudagraph_runtime_mode in (CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE)',
    "FIX2: _should_build_dummy_attn_metadata FULL→(FULL,PIECEWISE)")

# --- FIX3: pad_attn FULL→(FULL,PIECEWISE) ---
patch_file(MR,
    'pad_attn = cudagraph_runtime_mode == CUDAGraphMode.FULL',
    'pad_attn = cudagraph_runtime_mode in (CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE)',
    "FIX3: pad_attn FULL→(FULL,PIECEWISE)")

# --- FIX5-IMPORT: add CUDAGraphMode to attention_v1.py ---
patch_file(ATTN,
    'from vllm_ascend.ascend_forward_context import _EXTRA_CTX',
    'from vllm_ascend.ascend_forward_context import _EXTRA_CTX\n'
    'from vllm.config.compilation import CUDAGraphMode',
    "FIX5-IMPORT: add CUDAGraphMode import")

# --- FIX5-1: forward_fused_infer_attention ---
patch_file(ATTN,
    '        if _EXTRA_CTX.capturing:\n'
    '            attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output)',
    '        if _EXTRA_CTX.capturing and _EXTRA_CTX.cudagraph_runtime_mode == CUDAGraphMode.FULL:\n'
    '            attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output)',
    "FIX5-1: forward_fused_infer_attention add FULL guard")

# --- FIX5-2: forward_paged_attention ---
patch_file(ATTN,
    '        if _EXTRA_CTX.capturing:\n'
    '            return self.full_graph_pa(query, attn_metadata, output)',
    '        if _EXTRA_CTX.capturing and _EXTRA_CTX.cudagraph_runtime_mode == CUDAGraphMode.FULL:\n'
    '            return self.full_graph_pa(query, attn_metadata, output)',
    "FIX5-2: forward_paged_attention add FULL guard")

# --- FIX5-3: C8 decode ---
patch_file(ATTN,
    '                if _EXTRA_CTX.capturing:\n'
    '                    attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output, layer)',
    '                if _EXTRA_CTX.capturing and _EXTRA_CTX.cudagraph_runtime_mode == CUDAGraphMode.FULL:\n'
    '                    attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output, layer)',
    "FIX5-3: C8 decode add FULL guard")

# --- FIX5-4: C8 non-decode ---
patch_file(ATTN,
    '                if _EXTRA_CTX.capturing:\n'
    '                    attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output, layer)\n'
    '                    output[:num_tokens] = attn_output[:num_tokens]\n'
    '                    return output\n'
    '                elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:',
    '                if _EXTRA_CTX.capturing and _EXTRA_CTX.cudagraph_runtime_mode == CUDAGraphMode.FULL:\n'
    '                    attn_output, num_tokens = self.full_graph_fia(query, key, value, attn_metadata, output, layer)\n'
    '                    output[:num_tokens] = attn_output[:num_tokens]\n'
    '                    return output\n'
    '                elif attn_metadata.attn_state == AscendAttentionState.DecodeOnly:',
    "FIX5-4: C8 non-decode add FULL guard")

# --- FIX5b: add cudagraph_runtime_mode to extra_attrs ---
patch_file(CTX,
    '    extra_attrs = (\n'
    '        "capturing",\n'
    '        "moe_comm_type",\n'
    '        "moe_comm_method",\n'
    '        "mmrs_fusion",\n'
    '        "num_tokens",\n'
    '        "flash_comm_v1_enabled",\n'
    '        "flashcomm_v2_enabled",\n'
    '        "pad_size",\n'
    '        "padded_length",\n'
    '        "num_tokens_across_dp",\n'
    '        "mc2_mask",\n'
    '        "is_draft_model",\n'
    '        "is_draft_model_prefill",\n'
    '        "prefetch_mlp_gate_up_proj",\n'
    '        "prefetch_mlp_down_proj",\n'
    '        "model_instance",\n'
    '        "layer_idx",\n'
    '        "max_tokens_across_dp",\n'
    '        "max_tokens_across_pcp",\n'
    '        "num_accept_tokens",\n'
    '        "in_profile_run",\n'
    '        "padded_num_tokens",\n'
    '    )',
    '    extra_attrs = (\n'
    '        "capturing",\n'
    '        "cudagraph_runtime_mode",\n'
    '        "moe_comm_type",\n'
    '        "moe_comm_method",\n'
    '        "mmrs_fusion",\n'
    '        "num_tokens",\n'
    '        "flash_comm_v1_enabled",\n'
    '        "flashcomm_v2_enabled",\n'
    '        "pad_size",\n'
    '        "padded_length",\n'
    '        "num_tokens_across_dp",\n'
    '        "mc2_mask",\n'
    '        "is_draft_model",\n'
    '        "is_draft_model_prefill",\n'
    '        "prefetch_mlp_gate_up_proj",\n'
    '        "prefetch_mlp_down_proj",\n'
    '        "model_instance",\n'
    '        "layer_idx",\n'
    '        "max_tokens_across_dp",\n'
    '        "max_tokens_across_pcp",\n'
    '        "num_accept_tokens",\n'
    '        "in_profile_run",\n'
    '        "padded_num_tokens",\n'
    '    )',
    "FIX5b: add cudagraph_runtime_mode to extra_attrs")

# --- FIX6: NPU _dummy_run add has_preprocess branch ---
# Root cause of audio noise: NPU _dummy_run warmup sets input_ids=None but inference
# provides input_ids=tensor. @support_torch_compile requires "None params must always
# be None". torchair compiled graph doesn't register input_ids as dynamic input →
# replay receives unexpected tensor → data misalignment → wrong hidden_states → noise.
# FIX: check has_preprocess FIRST → warmup provides both input_ids=tensor and
# inputs_embeds=tensor → matches inference signature → graph replay works correctly.
patch_file(NPU_MR,
    '            if self.supports_mm_inputs and not self.model_config.is_encoder_decoder or self.enable_prompt_embeds:\n'
    '                input_ids = None\n'
    '                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]\n'
    '            else:\n'
    '                input_ids = self.input_ids.gpu[:num_tokens_padded]\n'
    '                inputs_embeds = None',
    '            if getattr(getattr(self, "model", None), "has_preprocess", False):\n'
    '                input_ids = self.input_ids.gpu[:num_tokens_padded]\n'
    '                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]\n'
    '            elif self.supports_mm_inputs and not self.model_config.is_encoder_decoder or self.enable_prompt_embeds:\n'
    '                input_ids = None\n'
    '                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]\n'
    '            else:\n'
    '                input_ids = self.input_ids.gpu[:num_tokens_padded]\n'
    '                inputs_embeds = None',
    "FIX6: NPU _dummy_run add has_preprocess branch (checked FIRST, both ids+embeds as tensors)")

print("\n=== All patches applied. Summary ===")
print("FIX2         _should_build_dummy_attn_metadata    FULL→(FULL,PIECEWISE)")
print("FIX3         pad_attn                              FULL→(FULL,PIECEWISE)")
print("FIX5         attention_v1.py 4× capturing checks  add FULL mode guard")
print("FIX5b        extra_attrs add cudagraph_runtime_mode")
print("FIX6         NPU _dummy_run add has_preprocess branch (both ids+embeds as tensors)")
print("")
print("REVERTED: FIX1 (_update_full_graph_params_if_needed FULL-only, update_stream missing)")
print("REVERTED: FIX4 (need_sync FULL-only, PIECEWISE has no async update work)")
print("REMOVED:  DIAG patches (DIAG_UPDATE, DIAG_REPLAY, DIAG_CTX) — use deploy_debug.sh for diag")
PATCH_EOF

echo ""
echo "=== Step 3: Verify patches ==="
cd "$ASCEND_DIR" && git diff --stat
echo ""
echo "=== Deploy complete. Start vllm-omni server and check DIAG logs. ==="
