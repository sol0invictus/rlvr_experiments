"""Check that the GRPO trainer received the YAML's hyperparameters (three paper-era ones once never reached TRL).

    python scripts/vm/check_effective_config.py <output_dir>/effective_config.json

Reads the file train_grpo.py writes (the YAML next to every GRPOConfig field) and exits 1 on any mismatch."""
import json
import sys

# GRPOConfig field -> (YAML section, key)
FIELDS = {
    "learning_rate": ("training", "learning_rate"),
    "loss_type": ("training", "loss_type"),
    "beta": ("training", "beta"),
    "epsilon_high": ("training", "epsilon_high"),
    "max_grad_norm": ("training", "max_grad_norm"),
    "lr_scheduler_type": ("training", "lr_scheduler_type"),
    "gradient_accumulation_steps": ("training", "gradient_accumulation_steps"),
    "per_device_train_batch_size": ("training", "per_device_train_batch_size"),
    "mask_truncated_completions": ("training", "mask_truncated_completions"),
    "max_steps": ("training", "max_steps"),
    "max_completion_length": ("generation", "max_completion_length"),
    "num_generations": ("generation", "num_generations"),
    "generation_batch_size": ("generation", "generation_batch_size"),
    "temperature": ("generation", "temperature"),
    "seed": (None, "seed"),
}


def norm(v):
    if hasattr(v, "value"):
        v = v.value
    if isinstance(v, str):
        v = v.split(".")[-1].lower()   # enum reprs like 'SchedulerType.CONSTANT'
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


def main(path):
    d = json.load(open(path))
    y, g = d["yaml"], d["grpo_config"]
    bad = []
    for field, (sec, key) in FIELDS.items():
        src = y if sec is None else y.get(sec, {})
        if key not in src:
            continue
        if field == "max_steps" and g.get(field) != src[key]:
            print(f"  note: max_steps {g.get(field)} (YAML {src[key]}): a CLI override, e.g. the smoke run")
            continue
        want, got = norm(src[key]), norm(g.get(field))
        print(f"  {field:30s} yaml={want!r:12} trainer={got!r}")
        if want != got:
            bad.append(field)
    if bad:
        print(f"MISMATCH: {bad}")
        sys.exit(1)
    print("effective config OK")


if __name__ == "__main__":
    main(sys.argv[1])
