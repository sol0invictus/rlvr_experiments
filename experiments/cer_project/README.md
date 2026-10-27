# Competence-Edge Routing (CER)

Per-token loss routing between SFT and RL based on measured competence.

## Phases

1. **Phase 1**: Instrumented GRPO — validate the inverted-U (gradient signal concentrates at intermediate-competence tokens)
2. **Phase 2**: CER training — route tokens to SFT/RL/skip based on entropy
3. **Phase 3**: Baselines and comparison

## Running

```bash
# Phase 1: Instrumented GRPO (validate premise)
torchrun --standalone --nproc_per_node=8 phase1_instrumented_grpo.py configs/phase1_qwen15b_gsm8k.yaml

# Phase 2: CER training
torchrun --standalone --nproc_per_node=8 train_cer.py configs/cer_qwen15b_gsm8k.yaml
```
