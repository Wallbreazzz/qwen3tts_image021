# qwen3tts_image021
先拉取三个组件代码，报错vllm-omni的cherry-pick，按顺序执行即可
```
cd /vllm-workspace/vllm && git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*" && git fetch origin && git checkout v0.21.0 && cd ..
cd /vllm-workspace/vllm-ascend && git remote add gcanlin https://github.com/gcanlin/vllm-ascend.git && git fetch gcanlin && git checkout cann85-with-v0.21.0-adapt && cd ..
git clone https://github.com/vllm-project/vllm-omni.git
cd /vllm-workspace/vllm-omni && git checkout d4c1395 && git fetch origin pull/3609/head:pr-3609 && git cherry-pick fa8b15f4 && git cherry-pick 0fc4032a
```
然后打一个patch脚本，脚本在本仓库中，用来开启piecewise全入图模式
```
bash deploy_patch.sh
```
然后打个补丁
```
python3 << 'PATCH_EOF'
filepath = '/vllm-workspace/vllm-omni/vllm_omni/platforms/npu/worker/npu_model_runner.py'
with open(filepath, 'r') as f:
    content = f.read()
old_sig = "def _talker_mtp_forward(self, decode_req_ids: list[str], inputs_embeds: torch.Tensor) -> None:"
new_sig = "def _talker_mtp_forward(\n        self,\n        decode_req_ids: list[str],\n        inputs_embeds: torch.Tensor,\n        start_offsets: list[int] | None = None,\n    ) -> None:"
content = content.replace(old_sig, new_sig)
old_loop = """        for idx, req_id in enumerate(decode_req_ids):
            req_index = self.input_batch.req_ids.index(req_id)
            start_offset = int(self.query_start_loc.cpu[req_index])
            inputs_embeds[start_offset : start_offset + 1] = req_embeds[idx : idx + 1]"""
new_loop = """        if start_offsets is None:
            start_offsets = []
            for req_id in decode_req_ids:
                req_index = self.input_batch.req_ids.index(req_id)
                start_offsets.append(int(self.query_start_loc.cpu[req_index]))
        for idx, (req_id, start_offset) in enumerate(zip(decode_req_ids, start_offsets, strict=True)):
            inputs_embeds[start_offset : start_offset + 1] = req_embeds[idx : idx + 1]"""
content = content.replace(old_loop, new_loop)
with open(filepath, 'w') as f:
    f.write(content)
print('Patch applied successfully')
PATCH_EOF
```
然后打第二个补丁
```
sed -i 's/from vllm_omni.platforms.npu.models.patch_qwen3_tts_code2wav import/from vllm_omni.platforms.npu.models.qwen3_tts_code2wav import/' /vllm-workspace/vllm-omni/vllm_omni/platforms/npu/platform.py
```
最后按顺序安装三个组件即可
```
cd /vllm-workspace/vllm && VLLM_TARGET_DEVICE=empty pip install -e . --no-build-isolation
cd /vllm-workspace/vllm-ascend && pip install -e .
pip install onnxruntime-cann>=1.23.2 \
    av>=14.0.0 omegaconf>=2.3.0 diffusers==0.38.0 \
    safetensors>=0.8.0rc0 accelerate==1.12.0 \
    soundfile>=0.13.1 cache-dit==1.3.0 tqdm>=4.66.0 \
    torchsde>=0.2.6 openai-whisper>=20250625 \
    "imageio[ffmpeg]>=2.37.2" x-transformers>=2.12.2 \
    einops>=0.8.1 prettytable>=3.8.0 aenum==3.1.16 \
    pyzmq>=25.0.0 janus>=1.0.0 pydub
cd /vllm-workspace/vllm-omni && VLLM_OMNI_TARGET_DEVICE=npu pip install -e ".[npu]" --no-build-isolation
```
