import pytest

from pl_keyboard import arch, tokenizer_spec


def test_tiers_present():
    assert set(arch.TIERS) == {"low", "medium", "high"}


def test_vocab_size_matches_tokenizer_spec():
    # The model's embedding/lm_head vocab MUST equal the tokenizer's piece count,
    # or the exported GGUF mismatches the tokenizer. CLAUDE.md flags this as a
    # "keep in sync" risk; pin it so a drift in either constant fails CI.
    assert arch.VOCAB_SIZE == tokenizer_spec.VOCAB_SIZE


def test_tier_config_merges_preset_and_fixed_fields():
    cfg = arch.tier_config("low")
    assert cfg["hidden_size"] == 512
    assert cfg["num_hidden_layers"] == 10
    assert cfg["num_attention_heads"] == 8
    assert cfg["intermediate_size"] == 2048
    # fixed across tiers
    assert cfg["max_position_embeddings"] == 256
    assert cfg["rope_theta"] == 10000.0
    assert cfg["rms_norm_eps"] == 1e-5
    assert cfg["hidden_act"] == "silu"
    assert cfg["attention_bias"] is False


def test_tier_config_unknown_raises():
    with pytest.raises(ValueError, match="unknown tier"):
        arch.tier_config("ultra")


def test_estimate_params_matches_expected_scale():
    low = arch.estimate_params(arch.tier_config("low"))
    med = arch.estimate_params(arch.tier_config("medium"))
    high = arch.estimate_params(arch.tier_config("high"))
    assert low < med < high
    assert 50e6 < low < 65e6  # ~57M
    assert 75e6 < med < 100e6  # ~86M
    assert 120e6 < high < 150e6  # ~136M


def test_autotune_large_vram_uses_max_batch():
    assert arch.autotune(24 * 1024**3, arch.tier_config("low")) == (64, 4)


def test_autotune_when_model_barely_fits_falls_back_to_batch_one():
    bs, accum = arch.autotune(512 * 1024**2, arch.tier_config("high"))
    assert bs == 1
    assert accum == 256


def test_device_choices_are_auto_cpu_cuda():
    assert arch.DEVICE_CHOICES == ("auto", "cpu", "cuda")


@pytest.mark.parametrize("cuda_available", [True, False])
def test_resolve_device_cpu_is_always_cpu(cuda_available):
    assert arch.resolve_device("cpu", cuda_available) == "cpu"


def test_resolve_device_auto_follows_availability():
    assert arch.resolve_device("auto", True) == "cuda"
    assert arch.resolve_device("auto", False) == "cpu"


def test_resolve_device_explicit_cuda_falls_back_to_cpu_when_unavailable():
    assert arch.resolve_device("cuda", True) == "cuda"
    assert arch.resolve_device("cuda", False) == "cpu"


@pytest.mark.parametrize("tier", ["low", "medium", "high"])
@pytest.mark.parametrize("gb", [4, 6, 8, 12, 16, 24, 48])
def test_autotune_invariants(tier, gb):
    bs, accum = arch.autotune(gb * 1024**3, arch.tier_config(tier))
    assert bs in (1, 2, 4, 8, 16, 32, 64)
    assert accum >= 1
    assert bs * accum >= arch.TARGET_EFFECTIVE_BATCH
