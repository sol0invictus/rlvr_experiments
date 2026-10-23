from .reasoning_gym_env import ReasoningGymEnvironment


def load_environment(config: dict):
    """Return the correct environment instance for the given config.

    Resolution order:
    1. ``environment.name == 'gsm8k'`` → GSM8KEnvironment (lazy import)
    2. ``environment.name == 'reasoning_gym'`` → ReasoningGymEnvironment
    3. Legacy fallback: infer gsm8k from ``data.dataset_name``
    """
    env_cfg = config.get("environment", {})
    name = env_cfg.get("name", "")

    if name == "gsm8k" or (
        not name and config.get("data", {}).get("dataset_name") == "openai/gsm8k"
    ):
        # Lazy import — math_verify only required when actually using GSM8K
        from .gsm8k import GSM8KEnvironment
        return GSM8KEnvironment(config)

    if name == "reasoning_gym":
        return ReasoningGymEnvironment(config)

    raise ValueError(
        f"Unknown environment '{name}'. "
        "Supported: 'gsm8k', 'reasoning_gym'."
    )
