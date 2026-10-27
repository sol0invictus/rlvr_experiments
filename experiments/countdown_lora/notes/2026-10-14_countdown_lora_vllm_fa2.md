# 2026-10-14/15 — Countdown LoRA+GRPO: vLLM + FlashAttention-2 migration

Change log for moving the Qwen3-4B countdown LoRA+GRPO run from HF `generate`
to vLLM rollouts with FlashAttention-2, plus every environment change made
along the way (including one accidental one).

## Result

| Setup | s / step | 200 steps |
|---|---|---|
| Before: HF generate, SDPA (miniconda base, TRL 0.29) | ~184 | ~10 h |
| After: vLLM colocate + FA2 (`.venv-vllm`, TRL 1.15) | ~35 | ~2 h |

3-step smoke test (`--max_steps 3`, output in the session scratchpad):
reward 0.66 / 0.59 / 0.13, clipped ratio 0.34 / 0.38 / 0.88. Step 3 was a hard
5-number puzzle; completions were coherent, just truncated at 768 tokens.

The first real run (HF generate) died at step 10/200 around 23:03 on 10-14,
with no traceback; most likely the Windows host went to sleep. Its leftovers
(checkpoint-2, adapter, train.log) are still in
`experiments/countdown_lora/results/grpo_countdown_lora_qwen3_4b/`.

## Code / config changes (all uncommitted)

### `train_grpo.py`
Earlier session:
- LoRA via `peft.LoraConfig`, driven by the `lora:` config block (r, alpha,
  dropout, target_modules, use_dora); passed as `peft_config` to `GRPOTrainer`.
- `model.attn_implementation` config key, passed through to `from_pretrained`.
- vLLM path: `use_vllm`, `vllm_mode`, `vllm_gpu_memory_utilization`,
  `vllm_enable_sleep_mode`, `vllm_max_model_length` from the `vllm:` block.
- `generation_batch_size` configurable (default `num_generations × per_device_bs`).
- `stop_strings` are applied only on the HF-generate path. The patched
  `_generate_single_turn` passes `tokenizer=` and writes EOS into the first pad
  slot, so the completion mask closes when pad ≠ eos. With vLLM both are skipped.

This session:
- transformers 5 removed `warmup_ratio`. The script now passes
  `warmup_steps=training_conf.get('warmup_steps', training_conf.get('warmup_ratio', 0))`;
  a value below 1 is read as a ratio, so old configs that set `warmup_ratio` still work.

### `callbacks/validation_callback.py` (earlier session)
- Batched validation generation (`batch_size` arg, default 16) with left
  padding, replacing the one-prompt-at-a-time loop.
- Passes `tokenizer=` to `generate()` (required when `stop_strings` is set).

### `experiments/countdown_lora/config.yaml` (new, untracked)
This session added:
```yaml
model:
  attn_implementation: kernels-community/flash-attn2   # hub FA2 kernel

vllm:
  enabled: true
  mode: colocate
  gpu_memory_utilization: 0.5   # fraction of TOTAL GPU mem: 8GB weights + ~4.7GB KV
  enable_sleep_mode: true       # frees vLLM memory during the training step
  max_model_length: 1280        # ~250 prompt + 768 completion
```
The rest of the recipe is unchanged: r=16, alpha=32, all-linear, lr 1e-5,
beta=0, dapo loss, 4 prompts × 8 gens, max_completion 768, 4-5 numbers,
targets 10-200.

### `.gitignore`
- Added `.venv-vllm/`.

## Environment changes

### `.venv-vllm` (project venv; use this for all work)
- Created 2026-10-14 22:50 with uv 0.10.2 on Python 3.12.2 (base interpreter
  `/root/miniconda3/bin`, no system site-packages).
- Key versions: torch 2.13.0+cu130, transformers 5.17.0, TRL 1.15.0 and
  vLLM 0.31.0 (both latest on PyPI), peft 0.21.2, accelerate 1.15.0,
  reasoning_gym 0.1.25.
- **This session:** `kernels` 0.12.3 → **0.16.2**, plus new deps kernels-data,
  sigstore, sigstore-models, sigstore-rekor-types, tuf, securesystemslib,
  rfc3161-client, rfc8785, id, pyopenssl, pyasn1, platformdirs, tomlkit.
  Nothing else was replaced.
  - Why: transformers 5.17 needs `kernels>=0.16,<0.17` for hub kernels.
- **The venv has no pip.** A bare `pip` resolves to miniconda base. Install with:
  ```bash
  uv pip install --python .venv-vllm/bin/python <pkg>
  ```

### FlashAttention-2: why the hub kernel instead of `flash-attn`
- There is no `flash-attn` wheel for torch 2.13/cu130, and the system nvcc is
  CUDA 11.5, too old to build one.
- `attn_implementation: kernels-community/flash-attn2` downloads a prebuilt
  kernel (cached in the HF hub cache).
- Checked against SDPA on a padded batch: same argmax at 100% of real tokens,
  max logit diff 0.28 (bf16 noise).
- vLLM uses its own attention backend for rollouts; FA2 here only affects the
  HF training forward/backward (and HF-generate validation).

### miniconda base (`/root/miniconda3`): ACCIDENTAL, currently broken
- A `pip install "kernels>=0.16.0,<0.17.0"` run with the venv activated went
  to base pip (the venv has none). It installed/upgraded 28 packages:
  anyio 4.15.1, cffi 2.1.1, click 8.5.0, cryptography 50.0.2, h11 0.16.0,
  hf_xet 1.7.0, httpcore2 2.13.1, httpx2 2.13.1, **huggingface_hub 2.2.0**,
  id 1.6.1, idna 3.20, kernels 0.16.2, kernels_data 0.16.2, platformdirs 4.13.0,
  pydantic 2.14.1, pydantic_core 2.50.1, pyjwt 2.15.1, pyopenssl 26.4.0,
  rfc3161_client 1.0.9, rfc8785 0.1.4, securesystemslib 1.5.1, sigstore 4.5.0,
  sigstore_models 0.0.6, sigstore_rekor_types 0.0.18, truststore 0.10.5,
  tuf 7.0.1, typing_extensions 4.16.0, typing_inspection 0.4.4.
  The previous versions were not recorded.
- **Effect:** `import transformers` in base now fails. Base transformers 5.2.0
  needs `huggingface_hub>=1.3,<2.0`.
- **Fix (dry-run checked; installs huggingface_hub 1.33.0 + h11 0.14.0):**
  ```bash
  /root/miniconda3/bin/python -m pip install "huggingface_hub>=1.3,<2.0"
  ```
- Base stack otherwise: torch 2.7.1, transformers 5.2.0, TRL 0.29.0,
  peft 0.14.0, vLLM 0.10.1.1.

## Known warnings
- `CUDA Error: invalid argument at /workspace/csrc/cumem_allocator.cpp:258`
  printed once during the smoke test. It comes from vLLM sleep mode's
  allocator under WSL. Training continued and outputs were fine. If it ever
  becomes fatal, set `vllm.enable_sleep_mode: false` and lower
  `gpu_memory_utilization`.

## Before the next full run
1. Disable sleep on the Windows host; launch with `nohup`.
2. Decide what to do with the dead run in `experiments/countdown_lora/results/grpo_countdown_lora_qwen3_4b/`
   (archive as `_hf_generate` or overwrite).
3. Launch:
   ```bash
   source .venv-vllm/bin/activate
   nohup python train_grpo.py experiments/countdown_lora/config.yaml \
     > experiments/countdown_lora/results/grpo_countdown_lora_qwen3_4b.log 2>&1 &
   ```
