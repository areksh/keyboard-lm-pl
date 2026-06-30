# keyboard-lm-pl

Wyszkol **polski** model językowy przewidujący następne słowo / model językowy oparty na gestach przesuwania dla [klawiatury FUTO](https://keyboard.futo.org),
dostarczany w formacie GGUF, który można zaimportować bezpośrednio na urządzenie. FUTO dostarcza transformator obsługujący wyłącznie język angielski; niniejszy projekt
tworzy jego polski odpowiednik, zainicjowany przez
[zgłoszenie nr 1212](https://github.com/futo-org/android-keyboard/issues/1212).

Język polski charakteryzuje się bogatą fleksją, więc przewidywanie oparte wyłącznie na słowniku nie pozwala wybrać prawidłowej deklinacji
(wpisanie `Stanów ` powinno przewidzieć `Zjednoczonych`). Transformer rozwiązuje ten problem — a co najważniejsze,
przywraca znaki diakrytyczne z tekstu wpisanego w alfabecie łacińskim: po wpisaniu `lozko` otrzymujemy `łóżko`.

> **Źródło i przypisanie.** Projekt ten opiera się na podejściu przedstawionym w
> [keyboard-lm-de](https://github.com/jblechert/keyboard-lm-de) autorstwa jblechert (wykorzystanym jako
> punkt odniesienia, a nie skopiowanym — język polski *odwraca* działanie filtrów znaków diakrytycznych) i implementuje
> [zgłoszenie nr 1212](https://github.com/futo-org/android-keyboard/issues/1212). Podobnie jak w przypadku modelu niemieckiego,
> podczas całego procesu tworzenia intensywnie wykorzystywano [Claude](https://www.anthropic.com/claude).

> **Status.** Zbudowany z wykorzystaniem podejścia „test-first”, **100% pokrycia linii i gałęzi, 185 testów.** Pełny potok jest już
> gotowy: pobieranie (`01`), czyszczenie (`02`), synteza (`03`), tokenizator (`04`/`04b`), XBU (`05`),
> szkolenie (`06`), konwersja (`07`), kwantyzacja (`08`), ocena (`09`). Rzeczywiste (bardzo małe) uruchomienie procesu „szkolenie → konwersja” zostało
> sprawdzone od początku do końca pod kątem generowania GGUF zgodnego z klawiaturą (test dymny integracji), prawdziwy model SentencePiece
> został zweryfikowany pod kątem spełnienia umowy ciągłości `<CHAR_*>`, a zestaw testów ewaluacyjnych jest uruchamiany
> w środowisku produkcyjnym na niewielkim modelu. Pozostałe prace obejmują dostrojenie i pełnowymiarowe szkolenie; nie jest wymagany żaden nowy kod.

## Główna idea: składanie znaków diakrytycznych

Ścieżka autokorekty/przesuwania w FUTO rozpoznaje wyłącznie podstawowe znaki łacińskie `<CHAR_A>..<CHAR_Z>`
(`TrainingDataGenerator.kt: permittedCharacters = ‘a-z'-’`). W układzie klawiatury polskiej wpisuje się litery podstawowe
(znaki diakrytyczne poprzez długie naciśnięcie), a przesuwanie palcem odwzorowuje litery podstawowe. Dlatego trenujemy model na:

```
dane wejściowe  (tokeny CHAR, po złożeniu):  l o z k o
wartość docelowa  (z zachowanymi znakami diakrytycznymi): łóżko
```

`ą→a ć→c ę→e ł→l ń→n ó→o ś→s ź→z ż→z`. Słowa z znakami diakrytycznymi **nigdy nie są pomijane** (niemieckie
repozytorium referencyjne je pomija — co jest fatalne w przypadku języka polskiego). Zobacz `pl_keyboard/diacritics.py` oraz `pl_keyboard/xbu.py`.

## Metodologia: TDD

Ścisłe przestrzeganie zasady **czerwony → zielony → refaktoryzacja**, **wymóg 100% pokrycia** (`--cov-fail-under` jest egzekwowane w CI).
Projekt ma na celu zapewnienie tego:

- **`pl_keyboard/`** — czysta biblioteka o niewielkiej liczbie zależności (brak `torch`/`datasets` przy importowaniu). Cała
  logika krytyczna dla formatu oraz logika polska znajduje się tutaj, dzięki czemu jest szybka i dokładnie przetestowana jednostkowo.
- **Interfejsy CLI `01…09`** — cienkie opakowania: wyłącznie parsowanie argumentów i operacje wejścia/wyjścia, przekazujące decyzje do biblioteki.
- **Ciężkie elementy** (zbiory danych HuggingFace `datasets`, Ollama, `llama-quantize`, uczenie na GPU) znajdują się za
  interfejsami umożliwiającymi wstrzykiwanie danych, które można testować przy użyciu danych testowych, a także za pomocą pojedynczego testu sprawdzającego działanie od początku do końca na procesorze.

## Rozwój

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -e ‘.[dev]’

pytest --cov=pl_keyboard --cov-branch --cov-report=term-missing   # 100% gate
ruff check . && ruff format --check .
```

Katalog `pl_keyboard/` nie wymaga żadnych zależności związanych z uczeniem maszynowym. Trenowanie/eksport (dodatkowy moduł `[train]`: torch, transformers,
datasets, sentencepiece, gguf) wymaga Pythona w wersji 3.10–3.12.

Testy integracyjne (`tests/integration/`) wykonują rzeczywisty proces trenowania→konwersji→oceny na procesorze na niewielkiej konfiguracji w
ciągu kilku sekund. Aby zamiast tego przetestować rzeczywistą warstwę architektury, należy przekazać opcję `--tier` (bardziej wymagające — ręczna kontrola typu end-to-end,
niebędąca częścią bramki):

```bash
pip install -e ‘.[dev,train]’
pytest tests/integration --tier low      # lub medium / high; pominąć w przypadku niewielkiej konfiguracji testowej
```

## Ścisłe wymagania (weryfikowane na podstawie kodu źródłowego klawiatury)

Plik `.gguf` zostanie odrzucony / wyświetlony jako „(Nieobsługiwany)”, chyba że spełnia poniższe wymagania — każde z nich jest testem wykonywalnym
w katalogu `pl_keyboard/`:

| Wymaganie | Moduł |
|---|---|
| Metadane `keyboardlm.languages` / `.features` / `.ext_tokenizer_type=sentencepiece` | `gguf_meta.py` |
| funkcje ⊆ zestaw obsługiwanych (`opt_*`/`_*` dopuszczalne) | `gguf_meta.py` |
| SentencePiece, `treat_whitespace_as_suffix`, znaki specjalne `<XBU><XBC><XEC><CHAR_A..Z>` | `tokenizer_spec.py`, `tokens.py` |
| **`<CHAR_A>.. <CHAR_Z>` przy 26 kolejnych identyfikatorach** (kod natywny zakłada to) | `tokenizer_spec.verify_special_tokens` |
| format linii szkoleniowej `… <XBU><CHAR_…><XBC>prawda <XEC>` (spacja po wartości „prawda”, brak spacji po tagach) | `tokens.format_word_correction` |

## Potok

Ponumerowane skrypty CLI obsługujące testowaną bibliotekę:

| Etap | Skrypt | Biblioteka, którą obsługuje | Status |
|---|---|---|---|
| pobieranie | `01_download.py` | `sources` (strumieniowanie HF) | ✓ |
| czyszczenie | `02_clean_training_data.py` | `cleaning.clean_line` | ✓ |
| synthetic | `03_generate_synthetic.py` | `synthetic` (klient Ollama, wstrzyknięty) | ✓ |
| tokenizer | `04_train_tokenizer.py` / `04b_check` | `tokenizer_spec` | ✓ |
| dane XBU | `05_make_xbu_data.py` | `xbu.augment_line` | ✓ |
| szkolenie | `06_train_model.py` | `arch`, `datamix` (ręczna pętla w Torch) | ✓ |
| konwersja | `07_convert_to_gguf.py` | `gguf_meta` | ✓ |
| kwantyzacja | `08_quantize.py` | `llama-quantize` | ✓ |
| ocena | `09_eval.py` | `evaluation` (wstrzyknięty model) | ✓ |
| informacje o wydaniu | `10_release_notes.py` | `evaluation`, `release_notes` (wstrzyknięty model + `llama-bench`) | ✓ |

Przykład (z środowiskiem `[train]` venv w Pythonie 3.10–3.12):

```bash
python 01_download.py --source c4 --output raw/c4.txt --limit 2000000   # strumieniowe pobieranie źródła HF
python 03_generate_synthetic.py --output raw/synthetic.txt --per-topic 50   # wymaga lokalnego Ollama
python 02_clean_training_data.py --input raw/*.txt --output data/clean.txt
python 04_train_tokenizer.py --input data/clean.txt --model-prefix data/tok/pl
python 04b_check_tokenizer.py --model data/tok/pl.model        # ograniczenie ciągłości
python 05_make_xbu_data.py --input data/clean.txt --output data/xbu.txt
python 06_train_model.py --input data/clean.txt data/xbu.txt --sp-model data/tok/pl.model \
    --tier medium --output-dir models/pl            # automatycznie dostosowuje wielkość partii do Twojej pamięci VRAM
python 07_convert_to_gguf.py --model-dir models/pl --sp-model data/tok/pl.model --output pl-f16.gguf
python 08_quantize.py --input pl-f16.gguf           # -> pl-Q3_K_M/Q4_0/Q6_K/Q8_0.gguf
python 09_eval.py --model-dir models/pl --sp-model data/tok/pl.model --eval-file data/heldout.txt
python 10_release_notes.py --model-dir models/pl --sp-model data/tok/pl.model \
    --eval-file data/heldout.txt --version v0.1.0 --steps 200000 --pre-release \
    --gguf 'pl-*.gguf' --output RELEASE.md --report metrics/v0.1.0.json
```

## Obserwowanie przebiegu: postęp i poziom szczegółowości

Każdy krok akceptuje opcję `--loglevel {none|error|warning|info|debug|dev}` (domyślnie `info`, uporządkowane
od najmniejszego do największego poziomu szczegółowości). Powolne kroki (`train`, `clean`, `make_xbu`, `download`, `quantize`,
`convert`) wyświetlają pasek postępu [tqdm](https://tqdm.github.io/) przy poziomie `info`+ — pasek treningu zawiera
na bieżąco aktualizowane wartości straty, szybkości i przewidywanego czasu zakończenia (ETA), dzięki czemu można sprawdzić, czy proces działa. Opcja `none` wyłącza zarówno pasek, jak i logi,
ale nadal wyświetla końcowy wynik; opcja `debug` dodaje szczegółowe informacje o poszczególnych plikach oraz okresowe dane; `dev` (o jeden poziom
niżej od `debug`) to strumień danych dla każdego kroku. Wiersze wyników trafiają do stdout, a logi/paski do stderr.

## Korzystanie z karty graficznej

Plik `06_train_model.py` przyjmuje opcję `--device {auto|cpu|cuda}` (domyślnie `auto`: karta graficzna, jeśli jest dostępna, w przeciwnym razie procesor). Program
trenujący przenosi model i każdą partię danych na to urządzenie oraz rejestruje komunikat `device=… (cuda_available=…)` podczas
uruchamiania, dzięki czemu można to sprawdzić na pierwszy rzut oka (a narzędzie `nvidia-smi` powinno wówczas wyświetlić ten proces). **Aby korzystać z GPU,
muszą być spełnione dwie warunki:**

1. **Kompilacja torch z obsługą CUDA.** W systemie Linux zwykłe polecenie `pip install -e „.[train]”` domyślnie pobiera już **kompilację z obsługą CUDA
  ** — pakiet wheel zawiera środowisko uruchomieniowe CUDA, więc nie trzeba samodzielnie instalować CUDA, a
  większość konfiguracji jest gotowa do pracy z GPU od razu po instalacji. Wersja **tylko na procesor CPU** (`torch …+cpu`, dla której
   `torch.cuda.is_available()` zawsze zwraca `False`) jest *włączana* poprzez indeks CPU
   (`--index-url https://download.pytorch.org/whl/cpu`) — w ten sposób jest kompilowane repozytorium deweloperskie `.venv`
   , aby zachować lekkość, i nie może korzystać z GPU. Sprawdź swoją wersję za pomocą
   `python -c „import torch; print(torch.__version__, torch.cuda.is_available())”`; jeśli wyświetli się
   wersja `+cpu`, zainstaluj ponownie pakiet CUDA pasujący do twojego sterownika, np.
   `pip install torch --index-url https://download.pytorch.org/whl/cu124`.
2. **`--device auto` (lub `cuda`).** Jeśli dostępna jest kompilacja Torch z obsługą CUDA, opcja `auto` wybiera procesor graficzny. Przekazanie
   `--device cuda` *bez* kompilacji CUDA powoduje wyświetlenie ostrzeżenia i przejście na procesor centralny zamiast awarii.

W trybie CUDA pętla uczy się z wykorzystaniem **autocastu bf16 (precyzja mieszana)**: mnożenia macierzowe i operacje uwagi są wykonywane w bf16 na
rdzeniach tensorowych, podczas gdy wagi główne, gradienty i stany algorytmu Adam pozostają w fp32 (bf16 wykorzystuje ten sam
zakres wykładników co fp32, więc nie jest potrzebne skalowanie bezstratne). Zmniejsza to zużycie pamięci na aktywacje o około połowę — co pozwala z grubsza podwoić
rozmiar partii — przy znikomym kosztem jakości, który i tak nie ma znaczenia, ponieważ model jest dostarczany w postaci skwantyzowanej.
W przypadku działania na procesorze (CPU) operacje pozostają w formacie fp32. Szacunek aktywacji w pliku `pl_keyboard/arch.py` jest skalibrowany pod kątem tego kosztu związanego z bf16.

## Dane syntetyczne (Ollama)

Krok `03` generuje syntetyczne polskie wiersze poprzez wywołanie **lokalnego serwera [Ollama](https://ollama.com)
** przez HTTP (`POST {host}/api/generate`). Ollama jest usługą zewnętrzną, a nie zależnością Pythona
— krok ten wykorzystuje wyłącznie bibliotekę standardową, więc nie potrzebujesz do niego środowiska `[train]`.

```bash
curl -fsSL https://ollama.com/install.sh | sh   # instaluje i uruchamia usługę na localhost:11434
ollama pull llama3.1                             # domyślny model --model
ollama list                                      # weryfikacja; lub: curl http://localhost:11434/api/tags
```

Następnie wykonaj krok `03`. Zastąp opcję `--model <nazwa>` dowolnym innym zainstalowanym modelem lub `--host <adres URL>`, aby
wskazać serwer zdalny/inny niż domyślny. Ten krok jest opcjonalny — pomiń go, jeśli trenujesz wyłącznie na
pobranych korpusach.

## Poziomy modelu

Architektura jest **do wyboru**, zależna od posiadanego procesora graficznego (nie ma narzuconego domyślnego ustawienia). Plik `06_train_model.py`
udostępnia opcję `--tier` i automatycznie dostosowuje parametry batch/grad-accum do pamięci VRAM (`pl_keyboard/arch.py`); zobacz
[Korzystanie z procesora graficznego](#using-your-gpu), aby wybrać urządzenie do szkolenia. Automatyczne dostosowanie limitów odbywa się w oparciu o
*wolnej* (a nie całkowitej) pamięci i rezerwuje miejsce na kontekst CUDA oraz fragmentację, więc np. karta o pojemności
8 GB przyjmuje wartości `batch=32 grad_accum=8`. Jeśli nadal występuje błąd OOM, należy nadpisać te wartości za pomocą wyraźnych opcji
`--batch-size N --grad-accum M` (należy zachować `N×M ≈ 256`, aby zachować efektywny rozmiar partii).

| Poziom | ukryte/warstwy/głowice/ffn | ~parametry | VRAM (bs64 / bs16+accum) | ~czas / 200 tys. kroków |
|---|---|---|---|---|
| niski | 512/10/8/2048 | ~57 mln | ~8–10 / ~6 GB | ~25–35 godz. |
| średni | 640/12/10/2560 | ~86 mln | ~12–14 / ~8 GB | ~35–45 godz. |
| wysoka | 768/12/12/3072 | ~136 mln | ~18–22 / ~10–12 GB | ~50–60 godz. |

Brak/słaba karta graficzna → chmura (Runpod/Vast, ~15–50 USD/uruchomienie). W kolejnych wydaniach będą publikowane gotowe, skwantyzowane pliki `.gguf`,
dzięki czemu większość użytkowników nigdy nie będzie musiała przeprowadzać treningu.

## Podziękowania i licencja

- Źródło (przeanalizowane, nie skopiowane): [keyboard-lm-de](https://github.com/jblechert/keyboard-lm-de) autorstwa jblechert.
- Kontrakt zweryfikowany względem [futo-org/android-keyboard](https://github.com/futo-org/android-keyboard).
- Kod: MIT (zobacz [`LICENSE`](LICENSE)).
- Wagi modelu (po opublikowaniu): [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/),
  podobnie jak w przypadku modelu niemieckiego — można z nich korzystać i udostępniać je bezpłatnie z podaniem źródła, wyłącznie do celów niekomercyjnych.
- Źródła danych szkoleniowych (fineweb-2/c4 ODC-By, OpenSubtitles, Wikipedia CC BY-SA) dokumentowane są w każdej wersji.

