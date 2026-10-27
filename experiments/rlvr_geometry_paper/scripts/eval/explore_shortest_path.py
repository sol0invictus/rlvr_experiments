import reasoning_gym
ds = reasoning_gym.create_dataset('shortest_path', size=10, seed=42)
for i, item in enumerate(ds):
    print(f"--- Sample {i} ---")
    print(f"Q: {item['question'][:500]}")
    print(f"A: {item['answer']}")
    m = item.get('metadata', {})
    print(f"Meta keys: {list(m.keys())}")
    for k, v in m.items():
        print(f"  {k}: {str(v)[:200]}")
    print()
