"""06_train_model: train a Llama keyboard model from scratch.

`main` (tier selection, arch overrides, VRAM auto-tune, I/O) is unit-tested with
the heavy bits mocked. The actual training loop (`_train_model`) and VRAM probe
(`_detect_vram`) are exercised for real by tests/integration/test_smoke.py.
"""

import argparse
import logging
import sys
from pathlib import Path

from cli import _runtime
from pl_keyboard import arch, logging_setup

log = logging.getLogger("pl_keyboard")


def _detect_vram() -> int:  # pragma: no cover - depends on GPU presence
    try:
        import torch

        if torch.cuda.is_available():
            # Free (not total) memory: the desktop/compositor and other
            # processes already hold some of the card, and autotune budgets
            # against what's actually available.
            free, _total = torch.cuda.mem_get_info()
            return free
    except Exception:
        pass
    return 0


def _cuda_available() -> bool:  # pragma: no cover - depends on torch build/GPU
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def _train_model(  # pragma: no cover - heavy torch training, covered by integration smoke
    *,
    config: dict,
    sp_model: str,
    inputs: list[str],
    steps: int,
    batch_size: int,
    grad_accum: int,
    output_dir: str,
    lr: float,
    seed: int,
    device: str,
) -> None:
    import random as _random

    import sentencepiece as spm
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM

    from pl_keyboard.datamix import mix, tokenize_and_chunk

    torch.manual_seed(seed)
    rng = _random.Random(seed)
    sp = spm.SentencePieceProcessor(model_file=sp_model)
    context_len = config["max_position_embeddings"]

    def line_iter(path):
        while True:
            produced = False
            with open(path, encoding="utf-8") as f:
                for line in f:
                    produced = True
                    yield line
            if not produced:
                return

    lines = mix([line_iter(p) for p in inputs], [1] * len(inputs), rng)
    chunks = tokenize_and_chunk(lines, lambda s: sp.encode(s, out_type=int), 1, 2, context_len)

    model = LlamaForCausalLM(
        LlamaConfig(vocab_size=sp.get_piece_size(), bos_token_id=1, eos_token_id=2, **config)
    )
    model.to(device)
    model.train()
    log.info(
        "training on %s: %s params, batch=%d x grad_accum=%d for %d steps",
        device,
        f"{model.num_parameters():,}",
        batch_size,
        grad_accum,
        steps,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    bar = _runtime.progress(range(steps), desc="train", log=log, total=steps, unit="step")
    for step in bar:
        opt.zero_grad()
        step_loss = 0.0
        for _ in range(grad_accum):
            ids = torch.tensor(
                [next(chunks) for _ in range(batch_size)], dtype=torch.long, device=device
            )
            loss = model(input_ids=ids, labels=ids).loss / grad_accum
            loss.backward()
            step_loss += loss.item()
        opt.step()
        bar.set_postfix(loss=f"{step_loss:.3f}")
        log.log(logging_setup.DEV, "step %d/%d loss=%.4f", step + 1, steps, step_loss)
        if (step + 1) % 100 == 0:
            log.debug("step %d/%d loss=%.4f", step + 1, steps, step_loss)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    log.info("saved model to %s", output_dir)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Train a Llama keyboard model.")
    p.add_argument("--input", nargs="+", required=True, help="Training text file(s).")
    p.add_argument("--sp-model", required=True, help="SentencePiece .model file.")
    p.add_argument("--output-dir", default="models/pl_keyboard")
    p.add_argument("--tier", choices=sorted(arch.TIERS), default="low")
    p.add_argument("--steps", type=int, default=200_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--vram-bytes", type=int, default=None, help="Override detected VRAM.")
    p.add_argument("--hidden", type=int, help="Override hidden_size.")
    p.add_argument("--layers", type=int, help="Override num_hidden_layers.")
    p.add_argument("--heads", type=int, help="Override num_attention_heads.")
    p.add_argument("--ffn", type=int, help="Override intermediate_size.")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument(
        "--device",
        choices=arch.DEVICE_CHOICES,
        default="auto",
        help="Where to train: auto (GPU if present), cpu, or cuda.",
    )
    _runtime.add_common_args(p)
    args = p.parse_args(argv)
    _runtime.configure(args)

    inputs = [f for f in args.input if Path(f).is_file()]
    if not inputs:
        print("no input files found", file=sys.stderr)
        return 1

    cuda = _cuda_available()
    device = arch.resolve_device(args.device, cuda)
    if args.device == "cuda" and not cuda:
        log.warning("CUDA requested but unavailable (CPU torch build or no GPU); using CPU instead")
    log.info("device=%s (cuda_available=%s)", device, cuda)

    config = arch.tier_config(args.tier)
    if args.hidden:
        config["hidden_size"] = args.hidden
    if args.layers:
        config["num_hidden_layers"] = args.layers
    if args.heads:
        config["num_attention_heads"] = args.heads
    if args.ffn:
        config["intermediate_size"] = args.ffn

    if args.batch_size and args.grad_accum:
        batch_size, grad_accum = args.batch_size, args.grad_accum
    else:
        vram = args.vram_bytes if args.vram_bytes is not None else _detect_vram()
        batch_size, grad_accum = arch.autotune(vram, config)

    print(f"tier={args.tier} batch={batch_size} grad_accum={grad_accum} steps={args.steps}")
    _train_model(
        config=config,
        sp_model=args.sp_model,
        inputs=inputs,
        steps=args.steps,
        batch_size=batch_size,
        grad_accum=grad_accum,
        output_dir=args.output_dir,
        lr=args.lr,
        seed=args.seed,
        device=device,
    )
    print(f"saved model to {args.output_dir}")
    return 0
