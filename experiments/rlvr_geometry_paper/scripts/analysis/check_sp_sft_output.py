import json
with open("/scratch/sunguha/rlvr_experiments/results/phase2_sp/sft_direct_300_eval/details_sft_direct_300_20260419_010003.json") as f:
    data = json.load(f)
results = data["shortest_path"]
for r in results[:5]:
    print(f"Expected: {r['ground_truth']}")
    print(f"Extracted: {r.get('extracted','NONE')}")
    print(f"Response ({len(r['response'])} chars): {r['response'][:250]}")
    print()

# Also check the GRPO-only best run (46%) for comparison
import glob
files = sorted(glob.glob("/scratch/sunguha/rlvr_experiments/results/phase2_sp/atomic_warm_300_eval/details_*.json"))
if files:
    with open(files[-1]) as f:
        data2 = json.load(f)
    results2 = data2["shortest_path"]
    correct2 = [r for r in results2 if r["correct"]]
    print("--- GRPO-Atomic-Warm-300 (46%) correct example ---")
    if correct2:
        r = correct2[0]
        print(f"Response ({len(r['response'])} chars): {r['response'][:300]}")
