_Polskie README jest [tutaj](README.pl.md) (przetłumaczone za pomocą DeepL, więc nie mogę zagwarantować, że tłumaczenie jest dobre)._

Train a **Polish** next-word / swipe language model for [FUTO Keyboard](https://keyboard.futo.org),
shipped as a GGUF you can import on-device. FUTO ships an English-only transformer; this project
builds the Polish equivalent, motivated by
[issue #1212](https://github.com/futo-org/android-keyboard/issues/1212).

Polish is heavily inflected, so dictionary-only prediction can't choose the right declension
(typing `Stanów ` should predict `Zjednoczonych`). A transformer fixes this — and, crucially, it
restores diacritics from base-latin typing: you type `lozko`, you get `łóżko`.

> **Reference & attribution.** This project follows the approach of
> [keyboard-lm-de](https://github.com/jblechert/keyboard-lm-de) by jblechert (studied as a
> reference, not copied — Polish *inverts* its diacritic filters) and implements
> [issue #1212](https://github.com/futo-org/android-keyboard/issues/1212). As with the German
> model, [Claude](https://www.anthropic.com/claude) was used extensively throughout development.

> **Status.** Built test-first, **100% line+branch coverage, 185 tests.** The full pipeline is in
> place: download (`01`), clean (`02`), synthetic (`03`), tokenizer (`04`/`04b`), XBU (`05`),
> train (`06`), convert (`07`), quantize (`08`), eval (`09`). A real (tiny) train → convert run is
> proven end-to-end to emit a keyboard-valid GGUF (integration smoke test), a real SentencePiece
> model is asserted to satisfy the `<CHAR_*>` contiguity contract, and the eval harness is run for
> real against a tiny model. Remaining work is tuning and a full-scale training run, not new code.

## The core idea: diacritic folding

FUTO's autocorrect/swipe path only knows base-latin `<CHAR_A>..<CHAR_Z>`
(`TrainingDataGenerator.kt: permittedCharacters = "a-z'-"`). On a Polish layout you type base
letters (diacritics via long-press) and swipe traces base letters. So we train the model on:

```
input  (CHAR tokens, folded):  l o z k o
truth  (kept with diacritics): łóżko
```

`ą→a ć→c ę→e ł→l ń→n ó→o ś→s ź→z ż→z`. Diacritic words are **never skipped** (the German
reference repo skips them — fatal for Polish). See `pl_keyboard/diacritics.py` and `pl_keyboard/xbu.py`.

## Methodology: TDD

Strict **red → green → refactor**, **100% coverage gate** (`--cov-fail-under` is enforced in CI).
The design exists to make that honest:

- **`pl_keyboard/`** — a pure, dependency-light library (no `torch`/`datasets` at import). All the
  format-critical and Polish logic lives here, so it's fast and exhaustively unit-tested.
- **CLIs `01…09`** — thin wrappers: argument parsing + I/O only, delegating decisions to the library.
- **Heavy edges** (HuggingFace `datasets`, Ollama, `llama-quantize`, GPU training) sit behind
  injectable interfaces that tests fake, plus one CPU end-to-end smoke test.

## Develop

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

pytest --cov=pl_keyboard --cov-branch --cov-report=term-missing   # 100% gate
ruff check . && ruff format --check .
```

`pl_keyboard/` needs no ML dependencies. Training/export (the `[train]` extra: torch, transformers,
datasets, sentencepiece, gguf) requires Python 3.10–3.12.

The integration tests (`tests/integration/`) run a real CPU train→convert→eval on a tiny config in
seconds. To exercise a real architecture tier instead, pass `--tier` (heavier — a manual end-to-end
check, not part of the gate):

```bash
pip install -e ".[dev,train]"
pytest tests/integration --tier low      # or medium / high; omit for the tiny smoke config
```

## The hard contract (verified from keyboard source)

A `.gguf` is rejected / shown "(Unsupported)" unless it satisfies these — each is an executable
test in `pl_keyboard/`:

| Requirement | Module |
|---|---|
| `keyboardlm.languages` / `.features` / `.ext_tokenizer_type=sentencepiece` metadata | `gguf_meta.py` |
| features ⊆ supported set (`opt_*`/`_*` tolerated) | `gguf_meta.py` |
| SentencePiece, `treat_whitespace_as_suffix`, `<XBU><XBC><XEC><CHAR_A..Z>` specials | `tokenizer_spec.py`, `tokens.py` |
| **`<CHAR_A>..<CHAR_Z>` at 26 consecutive ids** (native code assumes it) | `tokenizer_spec.verify_special_tokens` |
| training-line format `… <XBU><CHAR_…><XBC>truth <XEC>` (space after truth, none after tags) | `tokens.format_word_correction` |

## Pipeline

Numbered CLIs wrapping the tested library:

| Step | Script | Library it drives | Status |
|---|---|---|---|
| download | `01_download.py` | `sources` (HF streaming) | ✓ |
| clean | `02_clean_training_data.py` | `cleaning.clean_line` | ✓ |
| synthetic | `03_generate_synthetic.py` | `synthetic` (Ollama client, injected) | ✓ |
| tokenizer | `04_train_tokenizer.py` / `04b_check` | `tokenizer_spec` | ✓ |
| XBU data | `05_make_xbu_data.py` | `xbu.augment_line` | ✓ |
| train | `06_train_model.py` | `arch`, `datamix` (manual torch loop) | ✓ |
| convert | `07_convert_to_gguf.py` | `gguf_meta` | ✓ |
| quantize | `08_quantize.py` | `llama-quantize` | ✓ |
| eval | `09_eval.py` | `evaluation` (injected model) | ✓ |

Example (with a `[train]` venv on Python 3.10–3.12):

```bash
python 01_download.py --source c4 --output raw/c4.txt --limit 2000000   # stream a HF source
python 03_generate_synthetic.py --output raw/synthetic.txt --per-topic 50   # needs a local Ollama
python 02_clean_training_data.py --input raw/*.txt --output data/clean.txt
python 04_train_tokenizer.py --input data/clean.txt --model-prefix data/tok/pl
python 04b_check_tokenizer.py --model data/tok/pl.model        # contiguity contract
python 05_make_xbu_data.py --input data/clean.txt --output data/xbu.txt
python 06_train_model.py --input data/clean.txt data/xbu.txt --sp-model data/tok/pl.model \
    --tier medium --output-dir models/pl            # auto-tunes batch to your VRAM
python 07_convert_to_gguf.py --model-dir models/pl --sp-model data/tok/pl.model --output pl-f16.gguf
python 08_quantize.py --input pl-f16.gguf           # -> pl-Q3_K_M/Q4_0/Q6_K/Q8_0.gguf
python 09_eval.py --model-dir models/pl --sp-model data/tok/pl.model --eval-file data/heldout.txt
```

## Watching it run: progress & verbosity

Every step accepts `--loglevel {none|error|warning|info|debug|dev}` (default `info`, ordered
least→most verbose). The slow steps (`train`, `clean`, `make_xbu`, `download`, `quantize`,
`convert`) show a [tqdm](https://tqdm.github.io/) progress bar at `info`+ — training's bar carries a
live loss, rate, and ETA so you can tell it's working. `none` silences both the bar and the logs
while still printing the final result line; `debug` adds per-file / periodic detail; `dev` (one rung
below `debug`) is the per-step firehose. Result lines go to stdout, logs/bars to stderr.

## Using your GPU

`06_train_model.py` takes `--device {auto|cpu|cuda}` (default `auto`: GPU if present, else CPU). The
trainer moves the model and every batch onto that device and logs `device=… (cuda_available=…)` at
startup, so you can confirm at a glance (and `nvidia-smi` should then show the process). **Two things
must both hold to use the GPU:**

1. **A CUDA build of torch.** On Linux a plain `pip install -e ".[train]"` already pulls the **CUDA
   build** by default — the wheel bundles the CUDA runtime, so you don't install CUDA yourself and
   most setups are GPU-ready out of the box. The **CPU-only** build (`torch …+cpu`, for which
   `torch.cuda.is_available()` is always `False`) is *opt-in* via the CPU index
   (`--index-url https://download.pytorch.org/whl/cpu`) — that's how the repo's committed dev `.venv`
   is built, to stay lightweight, and it can't touch the GPU. Check yours with
   `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`; if it prints a
   `+cpu` version, reinstall a CUDA wheel matching your driver, e.g.
   `pip install torch --index-url https://download.pytorch.org/whl/cu124`.
2. **`--device auto` (or `cuda`).** With a CUDA torch build present, `auto` selects the GPU. Passing
   `--device cuda` *without* a CUDA build warns and falls back to CPU rather than crashing.

On CUDA the loop trains with **bf16 autocast (mixed precision)**: matmuls/attention run in bf16 on
the tensor cores while the master weights, gradients, and Adam states stay fp32 (bf16 shares fp32's
exponent range, so no loss scaling is needed). This ~halves activation memory — roughly doubling the
batch that fits — for negligible quality cost, which is moot anyway since the model ships quantized.
CPU runs stay fp32. The activation estimate in `pl_keyboard/arch.py` is calibrated to this bf16 cost.

## Synthetic data (Ollama)

Step `03` generates synthetic Polish lines by calling a **local [Ollama](https://ollama.com)
server** over HTTP (`POST {host}/api/generate`). Ollama is an external service, not a Python
dependency — the step uses only the stdlib, so you don't need the `[train]` venv for it.

```bash
curl -fsSL https://ollama.com/install.sh | sh   # installs + starts a service on localhost:11434
ollama pull llama3.1                             # the default --model
ollama list                                      # verify; or: curl http://localhost:11434/api/tags
```

Then run step `03`. Override `--model <name>` for any other installed model, or `--host <url>` to
point at a remote/non-default server. This step is optional — skip it if you only train on the
downloaded corpora.

## Model tiers

Architecture is **your choice**, driven by your GPU (no default imposed). `06_train_model.py`
exposes `--tier` and auto-tunes batch/grad-accum to your VRAM (`pl_keyboard/arch.py`); see
[Using your GPU](#using-your-gpu) for selecting the training device. Auto-tune budgets against
*free* (not total) memory and reserves headroom for the CUDA context + fragmentation, so e.g. an
8 GB card lands on `batch=32 grad_accum=8`. If you still hit an OOM, override with explicit
`--batch-size N --grad-accum M` (keep `N×M ≈ 256` to preserve the effective batch).

| Tier | hidden/layers/heads/ffn | ~params | VRAM (bs64 / bs16+accum) | ~time / 200k steps (desktop 4090) |
|---|---|---|---|---|
| low | 512/10/8/2048 | ~57M | ~8–10 / ~6 GB | ~25–35 h |
| medium | 640/12/10/2560 | ~86M | ~12–14 / ~8 GB | ~35–45 h |
| high | 768/12/12/3072 | ~136M | ~18–22 / ~10–12 GB | ~50–60 h |

The time column is anchored to a **desktop RTX 4090**. Weaker/laptop GPUs scale roughly linearly with
tensor throughput and are also pushed onto the small-batch + grad-accumulation path by limited VRAM,
so expect **~3–5× longer**: e.g. a **mobile 4070 (8 GB) runs the low tier in ~100 h**.

No/weak GPU → cloud (Runpod/Vast, ~$15–50/run). Releases will publish pre-built quantized `.gguf`s
so most users never train.

## Credits & licensing

- Reference (studied, not copied): [keyboard-lm-de](https://github.com/jblechert/keyboard-lm-de) by jblechert.
- Contract verified against [futo-org/android-keyboard](https://github.com/futo-org/android-keyboard).
- Code: MIT (see [`LICENSE`](LICENSE)).
- Model weights (when released): [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/),
  as with the German model — free to use and share with attribution, non-commercial.
- Training data sources (fineweb-2/c4 ODC-By, OpenSubtitles, Wikipedia CC BY-SA) documented per release.
