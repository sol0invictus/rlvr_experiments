from .reasoning_gym_env import ReasoningGymEnvironment
from .atomic_env import AtomicEnvironment


def load_environment(config: dict):
    """Return the correct environment instance for the given config.

    Resolution order:
    1. ``environment.name == 'gsm8k'`` → GSM8KEnvironment (lazy import)
    2. ``environment.name == 'reasoning_gym'`` → ReasoningGymEnvironment
    3. ``environment.name == 'atomic'`` → AtomicEnvironment
    4. ``environment.name == 'partial_credit_sp'`` / ``'feasible_only_sp'``
       → shortest-path reward variants (lazy import)
    5. Legacy fallback: infer gsm8k from ``data.dataset_name``
    """
    env_cfg = config.get("environment", {})
    name = env_cfg.get("name", "")

    if name == "gsm8k":
        from .gsm8k import GSM8KEnvironment
        return GSM8KEnvironment(config)

    if name == "reasoning_gym":
        return ReasoningGymEnvironment(config)

    if name == "atomic":
        return AtomicEnvironment(config)

    if name == "partial_credit_sp":
        from .partial_credit_sp_env import PartialCreditSPEnvironment
        return PartialCreditSPEnvironment(config)

    if name == "feasible_only_sp":
        from .feasible_only_sp_env import FeasibleOnlySPEnvironment
        return FeasibleOnlySPEnvironment(config)

    # Legacy path kept for backwards-compat with old configs
    if not name and config.get("data", {}).get("dataset_name") == "openai/gsm8k":
        from .gsm8k import GSM8KEnvironment
        return GSM8KEnvironment(config)

    raise ValueError(
        f"Unknown environment '{name}'. "
        "Supported: 'gsm8k', 'reasoning_gym', 'atomic', "
        "'partial_credit_sp', 'feasible_only_sp'."
    )
