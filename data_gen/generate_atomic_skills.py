"""
Unified Atomic Skill Data Generator

Generates L1/L2 atomic Q&A pairs for any (or all) reasoning-gym tasks
by parasitically extracting from task instances.

Usage:
    # Single task
    python data_gen/generate_atomic_skills.py --task countdown --num_instances 1000

    # All 10 tasks
    python data_gen/generate_atomic_skills.py --task all --num_instances 500

    # Custom output dir
    python data_gen/generate_atomic_skills.py --task maze --output_dir data/atoms/
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Add project root to path so imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_gen.atomic_extractors import EXTRACTORS


def generate_for_task(
    task_name: str,
    num_instances: int,
    seed: int,
    output_dir: Path,
) -> dict:
    """Generate atomic data for one task. Returns stats dict."""
    import reasoning_gym

    print(f"\n{'='*60}")
    print(f"  Task: {task_name}  |  Instances: {num_instances}  |  Seed: {seed}")
    print(f"{'='*60}")

    # Create reasoning-gym dataset
    try:
        dataset = reasoning_gym.create_dataset(task_name, size=num_instances, seed=seed)
    except Exception as e:
        print(f"  ERROR creating dataset: {e}")
        return {"task": task_name, "error": str(e)}

    # Extract atoms
    rng = random.Random(seed)
    extractor = EXTRACTORS[task_name]
    all_atoms = []
    skill_counts: dict[str, int] = {}

    for instance in dataset:
        try:
            atoms = extractor(instance, rng)
            for atom in atoms:
                atom["task"] = task_name  # tag with source task
            all_atoms.extend(atoms)
            for atom in atoms:
                skill_counts[atom["skill"]] = skill_counts.get(atom["skill"], 0) + 1
        except Exception as e:
            print(f"  WARN: extractor failed on instance: {e}")
            continue

    # Save
    output_path = output_dir / f"atoms_{task_name}.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for atom in all_atoms:
            f.write(json.dumps(atom) + "\n")

    # Print summary
    print(f"\n  Generated {len(all_atoms)} atomic Q&A pairs")
    print(f"  Output: {output_path}")
    print(f"\n  {'Skill':<30} {'Layer':<6} {'Count':>6}")
    print(f"  {'─'*30} {'─'*6} {'─'*6}")
    for skill, count in sorted(skill_counts.items()):
        layer = next(a["layer"] for a in all_atoms if a["skill"] == skill)
        print(f"  {skill:<30} {layer:<6} {count:>6}")
    print(f"  {'─'*30} {'─'*6} {'─'*6}")
    print(f"  {'TOTAL':<30} {'':6} {len(all_atoms):>6}")
    print(f"  Avg per instance: {len(all_atoms) / num_instances:.1f}")

    return {
        "task": task_name,
        "instances": num_instances,
        "total_atoms": len(all_atoms),
        "avg_per_instance": round(len(all_atoms) / num_instances, 1),
        "skills": skill_counts,
        "output": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Generate atomic skill data")
    parser.add_argument("--task", type=str, default="all",
                        help="Task name or 'all' (default: all)")
    parser.add_argument("--num_instances", type=int, default=500,
                        help="Number of reasoning-gym instances per task (default: 500)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="data_gen/data/atoms",
                        help="Output directory (default: data_gen/data/atoms)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.task == "all":
        tasks = list(EXTRACTORS.keys())
    else:
        if args.task not in EXTRACTORS:
            print(f"ERROR: Unknown task '{args.task}'")
            print(f"Available: {', '.join(EXTRACTORS.keys())}")
            return
        tasks = [args.task]

    print(f"Generating atomic data for {len(tasks)} task(s): {', '.join(tasks)}")

    all_stats = []
    for task in tasks:
        stats = generate_for_task(task, args.num_instances, args.seed, output_dir)
        all_stats.append(stats)

    # Grand summary
    print(f"\n{'='*60}")
    print(f"  GRAND SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Task':<20} {'Atoms':>8} {'Avg/Inst':>10}")
    print(f"  {'─'*20} {'─'*8} {'─'*10}")
    grand_total = 0
    for s in all_stats:
        if "error" in s:
            print(f"  {s['task']:<20} {'ERROR':>8}")
        else:
            print(f"  {s['task']:<20} {s['total_atoms']:>8} {s['avg_per_instance']:>10.1f}")
            grand_total += s["total_atoms"]
    print(f"  {'─'*20} {'─'*8} {'─'*10}")
    print(f"  {'TOTAL':<20} {grand_total:>8}")

    # Save summary JSON
    summary_path = output_dir / "generation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")


if __name__ == "__main__":
    main()
