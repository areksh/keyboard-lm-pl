"""GGUF keyboardlm.* metadata builder + the acceptance validator.

Mirrors the keyboard's checks so we fail fast in CI instead of discovering an
"(Unsupported)" model on the phone:
  - supported feature set: ModelPaths.kt:79-87
  - feature filter (opt_*/_* tolerated): ModelPaths.kt:157-163
  - isUnsupported(): ModelPaths.kt:46-48
The actual GGUF writing (and the binary ext_tokenizer_data blob) lives in the
converter CLI; this module is pure and fully unit-tested.
"""

import re
from collections.abc import Sequence

# ModelPaths.kt supportedFeatures
SUPPORTED_FEATURES: frozenset[str] = frozenset(
    {
        "base_v1",
        "inverted_space",
        "xbu_char_autocorrect_v1",
        "lora_finetunable_v1",
        "xc0_swipe_typing_v1",
        "char_embed_mixing_v1",
        "experiment_linear_208_209_210",
    }
)

# What we ship for the Polish v1 model (proven by the German model).
DEFAULT_FEATURES = "xbu_char_autocorrect_v1 char_embed_mixing_v1"

# HuggingFace Llama tensor name -> GGUF tensor name.
_DIRECT_TENSORS = {
    "model.embed_tokens.weight": "token_embd.weight",
    "model.norm.weight": "output_norm.weight",
    "lm_head.weight": "output.weight",
}
_LAYER_TENSORS = {
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
}
_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.(.+)$")


def hf_to_gguf_tensor_name(name: str) -> str | None:
    """Map a HuggingFace Llama tensor name to its GGUF name, or None to skip
    (e.g. rotary embedding buffers that aren't stored in GGUF)."""
    if name in _DIRECT_TENSORS:
        return _DIRECT_TENSORS[name]
    m = _LAYER_RE.match(name)
    if m:
        mapped = _LAYER_TENSORS.get(m.group(2))
        if mapped is not None:
            return f"blk.{m.group(1)}.{mapped}"
    return None


def keyboard_metadata(languages: Sequence[str], features: str = DEFAULT_FEATURES) -> dict[str, str]:
    """The string keyboardlm.* metadata entries written into the GGUF header."""
    return {
        "keyboardlm.languages": " ".join(languages),
        "keyboardlm.features": features,
        "keyboardlm.ext_tokenizer_type": "sentencepiece",
    }


def unsupported_features(features: Sequence[str]) -> list[str]:
    """Features the keyboard would reject (unknown, and not prefixed opt_/_)."""
    return [f for f in features if f not in SUPPORTED_FEATURES and not f.startswith(("opt_", "_"))]


def is_unsupported(features: Sequence[str], tokenizer_type: str, languages: Sequence[str]) -> bool:
    """Mirror of ModelInfo.isUnsupported()."""
    return not features or tokenizer_type == "None" or not languages


def validate(
    languages: Sequence[str],
    features_str: str,
    tokenizer_type: str = "sentencepiece",
) -> None:
    """Raise ValueError if this metadata would be rejected / Unsupported."""
    features = features_str.split()
    if is_unsupported(features, tokenizer_type, languages):
        raise ValueError("model would be Unsupported (empty features/languages or no tokenizer)")
    bad = unsupported_features(features)
    if bad:
        raise ValueError(f"unsupported features: {bad}")
