import json
from pathlib import Path

for size, prefix in [("3b","grpo_qwen25_3b"),("4b","grpo_qwen4b"),("7b","grpo_qwen25_7b"),("8b","grpo_qwen3_8b")]:
    d = Path(f"experiments/rlvr_geometry_paper/outputs/eval_{size}")

    # Baseline
    mf = sorted((d / "baseline").glob("metrics_*.json"))
    if mf:
        data = json.loads(mf[-1].read_text())
        ba = data["tasks"]["basic_arithmetic"]
        base_acc = ba["accuracy"]
        base_n = ba["num_correct"]
    else:
        base_acc = -1
        base_n = -1

    # GRPO arith model
    mn = f"{prefix}_basic_arithmetic"
    mf2 = sorted((d / mn).glob("metrics_*.json"))
    if mf2:
        data2 = json.loads(mf2[-1].read_text())
        ga = data2["tasks"]["basic_arithmetic"]
        grpo_acc = ga["accuracy"]
        grpo_n = ga["num_correct"]
        grpo_total = ga["num_samples"]
    else:
        grpo_acc = -1
        grpo_n = -1
        grpo_total = -1

    delta = (grpo_acc - base_acc) * 100
    print(f"{size}: baseline={base_acc:.0%} ({base_n}/50)  GRPO={grpo_acc:.0%} ({grpo_n}/{grpo_total})  delta={delta:+.0f}%")
