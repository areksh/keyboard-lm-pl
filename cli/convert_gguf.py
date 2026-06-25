"""07_convert_to_gguf: HuggingFace Llama + SentencePiece -> keyboard GGUF.

`main` (arg parsing + the metadata acceptance check) is unit-tested with `_convert`
mocked; `_convert` itself does the heavy torch/gguf I/O and is exercised for real by
tests/integration/test_convert_smoke.py.
"""

import argparse
import logging
import sys
from pathlib import Path

from cli import _runtime
from pl_keyboard import gguf_meta

log = logging.getLogger("pl_keyboard")


def _convert(  # pragma: no cover - heavy torch/gguf I/O, covered by integration smoke test
    model_dir: str,
    sp_model: str,
    out_path: str,
    name: str,
    languages: list[str],
    features: str,
) -> None:
    import numpy as np
    import sentencepiece as spm
    import torch
    from gguf import GGMLQuantizationType, GGUFWriter, TokenType
    from transformers import LlamaConfig, LlamaForCausalLM

    config = LlamaConfig.from_pretrained(model_dir)
    model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    state = model.state_dict()

    sp = spm.SentencePieceProcessor(model_file=sp_model)
    tokens, scores, types = [], [], []
    for i in range(sp.get_piece_size()):
        tokens.append(sp.id_to_piece(i).encode("utf-8"))
        scores.append(sp.get_score(i))
        if sp.IsUnknown(i):
            types.append(TokenType.UNKNOWN)
        elif sp.IsControl(i):
            types.append(TokenType.CONTROL)
        elif sp.IsByte(i):
            types.append(TokenType.BYTE)
        else:
            types.append(TokenType.NORMAL)

    writer = GGUFWriter(out_path, arch="llama")
    writer.add_name(name)
    writer.add_context_length(config.max_position_embeddings)
    writer.add_embedding_length(config.hidden_size)
    writer.add_block_count(config.num_hidden_layers)
    writer.add_feed_forward_length(config.intermediate_size)
    writer.add_head_count(config.num_attention_heads)
    writer.add_head_count_kv(config.num_key_value_heads)
    writer.add_layer_norm_rms_eps(config.rms_norm_eps)
    writer.add_rope_freq_base(getattr(config, "rope_theta", 10000.0))
    writer.add_file_type(GGMLQuantizationType.F16)

    writer.add_tokenizer_model("llama")
    writer.add_token_list(tokens)
    writer.add_token_scores(scores)
    writer.add_token_types(types)
    writer.add_bos_token_id(config.bos_token_id or 1)
    writer.add_eos_token_id(config.eos_token_id or 2)
    writer.add_unk_token_id(0)
    writer.add_pad_token_id(3)

    for key, value in gguf_meta.keyboard_metadata(languages, features).items():
        writer.add_string(key, value)
    # Raw bytes -> GGUF serializes as a UINT8 array, which is what the keyboard
    # reads back for keyboardlm.ext_tokenizer_data.
    writer.add_array("keyboardlm.ext_tokenizer_data", Path(sp_model).read_bytes())

    for hf_name, tensor in _runtime.progress(
        state.items(), desc="tensors", log=log, total=len(state), unit="tensor"
    ):
        gguf_name = gguf_meta.hf_to_gguf_tensor_name(hf_name)
        if gguf_name is None:
            continue
        arr = tensor.float().numpy()
        if gguf_name.endswith("_norm.weight"):
            writer.add_tensor(gguf_name, arr.astype(np.float32), raw_dtype=GGMLQuantizationType.F32)
        else:
            writer.add_tensor(gguf_name, arr.astype(np.float16), raw_dtype=GGMLQuantizationType.F16)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert a trained Llama model to keyboard GGUF.")
    p.add_argument("--model-dir", required=True, help="HuggingFace checkpoint directory.")
    p.add_argument("--sp-model", required=True, help="SentencePiece .model file.")
    p.add_argument("--output", required=True, help="Output .gguf path.")
    p.add_argument("--languages", nargs="+", default=["pl"])
    p.add_argument("--features", default=gguf_meta.DEFAULT_FEATURES)
    p.add_argument("--name", default="pl_keyboard")
    _runtime.add_common_args(p)
    args = p.parse_args(argv)
    _runtime.configure(args)

    try:
        gguf_meta.validate(args.languages, args.features)
    except ValueError as e:
        print(f"refusing to write Unsupported model: {e}", file=sys.stderr)
        return 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    log.info("converting %s -> %s (languages=%s)", args.model_dir, out, ",".join(args.languages))
    _convert(args.model_dir, args.sp_model, str(out), args.name, args.languages, args.features)
    print(f"wrote {out}")
    return 0
