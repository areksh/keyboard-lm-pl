"""Model architecture tiers and VRAM auto-tuning.

Tier is an end-user choice driven by the GPU they have — no default is imposed.
These are pure functions (VRAM is an argument), so they are fully unit-testable
without a GPU. The training CLI calls `tier_config` + `autotune` and feeds the
result to the (injected) trainer.
"""

import math

VOCAB_SIZE = 15008  # matches the tokenizer (3 + 26 specials + learned pieces)
TARGET_EFFECTIVE_BATCH = 256

# Fixed across every tier (must match the keyboard's expectations / training).
FIXED_CONFIG: dict[str, object] = {
    "max_position_embeddings": 256,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-5,
    "hidden_act": "silu",
    "attention_bias": False,
}

# (hidden, layers, heads, ffn) — ~57M / ~86M / ~136M parameters.
TIERS: dict[str, dict[str, int]] = {
    "low": {
        "hidden_size": 512,
        "num_hidden_layers": 10,
        "num_attention_heads": 8,
        "intermediate_size": 2048,
    },
    "medium": {
        "hidden_size": 640,
        "num_hidden_layers": 12,
        "num_attention_heads": 10,
        "intermediate_size": 2560,
    },
    "high": {
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
    },
}

# Training-memory heuristics (bf16 weights + fp32 Adam states ~= 16 B/param).
_BYTES_PER_PARAM_TRAIN = 16
# Per-sample activation cost proxy: context * hidden * layers * this many bytes.
_ACT_BYTES_FACTOR = 64
_BATCH_CHOICES = (64, 32, 16, 8, 4, 2, 1)


def tier_config(name: str) -> dict:
    """Full model config for a named tier (preset merged with FIXED_CONFIG)."""
    if name not in TIERS:
        raise ValueError(f"unknown tier: {name!r} (choose from {sorted(TIERS)})")
    return {**TIERS[name], **FIXED_CONFIG}


def estimate_params(config: dict, vocab_size: int = VOCAB_SIZE) -> int:
    """Rough Llama parameter count (untied embeddings + lm_head)."""
    h = config["hidden_size"]
    layers = config["num_hidden_layers"]
    ffn = config["intermediate_size"]
    embeddings = 2 * vocab_size * h
    per_layer = 4 * h * h + 3 * h * ffn
    return int(embeddings + layers * per_layer)


def autotune(
    vram_bytes: int,
    config: dict,
    vocab_size: int = VOCAB_SIZE,
    target_effective_batch: int = TARGET_EFFECTIVE_BATCH,
) -> tuple[int, int]:
    """Pick (batch_size, grad_accum) that fit `vram_bytes` while keeping the
    effective batch >= `target_effective_batch`."""
    reserved = estimate_params(config, vocab_size) * _BYTES_PER_PARAM_TRAIN
    available = vram_bytes - reserved
    act_per_sample = (
        config["max_position_embeddings"]
        * config["hidden_size"]
        * config["num_hidden_layers"]
        * _ACT_BYTES_FACTOR
    )
    max_batch = max(1, available // act_per_sample)

    batch = next((b for b in _BATCH_CHOICES if b <= max_batch), 1)
    grad_accum = math.ceil(target_effective_batch / batch)
    return batch, grad_accum
