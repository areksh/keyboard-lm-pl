"""09_eval: score a trained model on Polish benchmarks + held-out perplexity.

`main` orchestrates the pure metrics in `pl_keyboard.evaluation` with model
inference injected as callables; `_load_model` builds those callables from a real
torch/transformers/sentencepiece model and is marked `# pragma: no cover`
(heavy), exercised for real by tests/integration/test_eval_smoke.py.
"""

import argparse
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path

from cli import _runtime
from pl_keyboard import evaluation

log = logging.getLogger("pl_keyboard")

# (restore folded word, predict next word, score line -> (sum_nll, n_tokens)).
Model = tuple[Callable[[str], str], Callable[[str], str], Callable[[str], tuple[float, int]]]


def _load_model(model_dir: str, sp_model: str) -> Model:  # pragma: no cover - heavy torch inference
    import torch
    from sentencepiece import SentencePieceProcessor
    from transformers import LlamaForCausalLM

    from pl_keyboard import tokens

    sp = SentencePieceProcessor(model_file=sp_model)
    model = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.eval()
    bos = model.config.bos_token_id or 1
    eos = model.config.eos_token_id or 2
    xbu, xbc, xec = (
        sp.piece_to_id(t)
        for t in (tokens.BEGIN_USER_INPUT, tokens.BEGIN_CORRECTION, tokens.END_CORRECTION)
    )

    def greedy(prompt_ids: list[int], max_new: int, stop: int) -> list[int]:
        ids = list(prompt_ids)
        for _ in range(max_new):
            with torch.no_grad():
                logits = model(input_ids=torch.tensor([ids])).logits
            nxt = int(logits[0, -1].argmax())
            if nxt in (stop, eos):
                break
            ids.append(nxt)
        return ids[len(prompt_ids) :]

    def restore(folded: str) -> str:
        char_ids = [sp.piece_to_id(f"<CHAR_{c.upper()}>") for c in folded if c.isalpha()]
        out = greedy([bos, xbu, *char_ids, xbc], max_new=12, stop=xec)
        return sp.decode(out).strip()

    def predict(context: str) -> str:
        out = greedy([bos, *sp.encode(context + " ", out_type=int)], max_new=6, stop=eos)
        decoded = sp.decode(out).strip().split(" ")
        return decoded[0] if decoded else ""

    def score(line: str) -> tuple[float, int]:
        ids = [bos, *sp.encode(line, out_type=int), eos]
        t = torch.tensor([ids])
        with torch.no_grad():
            loss = model(input_ids=t, labels=t).loss
        n_tokens = len(ids) - 1  # next-token targets
        return loss.item() * n_tokens, n_tokens

    return restore, predict, score


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate a trained keyboard model.")
    p.add_argument("--model-dir", required=True, help="HuggingFace checkpoint directory.")
    p.add_argument("--sp-model", required=True, help="SentencePiece .model file.")
    p.add_argument("--eval-file", default=None, help="Held-out text for perplexity (optional).")
    p.add_argument("--report", default=None, help="Write the JSON report here (optional).")
    _runtime.add_common_args(p)
    args = p.parse_args(argv)
    _runtime.configure(args)

    if not Path(args.model_dir).is_dir():
        print(f"model dir not found: {args.model_dir}", file=sys.stderr)
        return 1

    log.info("loading model from %s for evaluation", args.model_dir)
    restore, predict, score = _load_model(args.model_dir, args.sp_model)
    log.debug("running diacritic-restoration and next-word benchmarks")
    report: dict[str, object] = {
        "diacritic_restoration_accuracy": evaluation.accuracy(
            evaluation.DIACRITIC_BENCHMARK, restore
        ),
        "next_word_accuracy": evaluation.accuracy(evaluation.NEXT_WORD_BENCHMARK, predict),
        "perplexity": None,
    }
    if args.eval_file:
        lines = [
            line.strip()
            for line in Path(args.eval_file).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if lines:
            report["perplexity"] = evaluation.corpus_perplexity(lines, score)

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return 0
