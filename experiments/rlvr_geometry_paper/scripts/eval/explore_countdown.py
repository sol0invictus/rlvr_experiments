import reasoning_gym
ds = reasoning_gym.create_dataset('countdown', size=10, seed=42,
    min_numbers=4, max_numbers=5, min_target=10, max_target=200)
for i, item in enumerate(ds):
    print(f"--- Sample {i} ---")
    print(f"Q: {item['question']}")
    print(f"A: {item['answer']}")
    m = item.get('metadata', {})
    print(f"Meta: target={m.get('target')}, numbers={m.get('numbers')}")
    print()
