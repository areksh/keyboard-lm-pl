"""Integration-test fixtures: pick the architecture from --tier.

By default the integration tests use a tiny config that instantiates/trains in
seconds on CPU. Passing --tier {low,medium,high} runs them against the real
pl_keyboard.arch preset instead (heavier — for a manual end-to-end check):

    pytest tests/integration -p no:cacheprovider --tier low
"""

import pytest

# Tiny architecture used when no --tier is given (LlamaConfig kwargs).
_SMOKE_ARCH: dict[str, object] = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
}


@pytest.fixture
def tier(request) -> str | None:
    """The --tier requested, or None for the tiny smoke config."""
    return request.config.getoption("--tier")


@pytest.fixture
def arch_config(tier) -> dict:
    """LlamaConfig architecture kwargs: tiny smoke config by default, or the real
    tier preset (pl_keyboard.arch.tier_config) when --tier is passed."""
    if tier is None:
        return dict(_SMOKE_ARCH)
    from pl_keyboard import arch

    return arch.tier_config(tier)


@pytest.fixture
def arch_cli_args(tier) -> list[str]:
    """`06_train_model` flags selecting the architecture: tiny overrides by
    default, or just `--tier <tier>` (letting arch.tier_config drive it)."""
    if tier is None:
        return ["--tier", "low", "--hidden", "32", "--layers", "2", "--heads", "2", "--ffn", "64"]
    return ["--tier", tier]
