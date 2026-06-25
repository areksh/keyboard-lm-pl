from pl_keyboard import gguf_meta as gm


def test_direct_tensor_names():
    assert gm.hf_to_gguf_tensor_name("model.embed_tokens.weight") == "token_embd.weight"
    assert gm.hf_to_gguf_tensor_name("model.norm.weight") == "output_norm.weight"
    assert gm.hf_to_gguf_tensor_name("lm_head.weight") == "output.weight"


def test_per_layer_tensor_names():
    assert (
        gm.hf_to_gguf_tensor_name("model.layers.0.self_attn.q_proj.weight") == "blk.0.attn_q.weight"
    )
    assert (
        gm.hf_to_gguf_tensor_name("model.layers.5.mlp.down_proj.weight") == "blk.5.ffn_down.weight"
    )
    assert (
        gm.hf_to_gguf_tensor_name("model.layers.2.post_attention_layernorm.weight")
        == "blk.2.ffn_norm.weight"
    )


def test_unknown_tensor_names_return_none():
    assert gm.hf_to_gguf_tensor_name("model.layers.0.something_else.weight") is None
    assert gm.hf_to_gguf_tensor_name("model.rotary_emb.inv_freq") is None
