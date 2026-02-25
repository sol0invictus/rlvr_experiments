from .reasoning_gym_env import ReasoningGymEnvironment
from .gsm8k import GSM8KEnvironment


def load_environment(config: dict):
    """Return the correct environment instance for the given config.

    Resolution order:
    1. ``environment.name == 'gsm8k'`` → GSM8KEnvironment
    2. ``environment.name == 'reasoning_gym'`` → ReasoningGymEnvironment
    3. Legacy fallback: infer gsm8k from ``data.dataset_name``
    """
    env_cfg = config.get("environment", {})
    name = env_cfg.get("name", "")

    if name == "gsm8k":
        return GSM8KEnvironment(config)

    if name == "reasoning_gym":
        return ReasoningGymEnvironment(config)

    # Legacy path kept for backwards-compat with old configs
    if not name and config.get("data", {}).get("dataset_name") == "openai/gsm8k":
        return GSM8KEnvironment(config)

    raise ValueError(
        f"Unknown environment '{name}'. "
        "Supported: 'gsm8k', 'reasoning_gym'."
    )

