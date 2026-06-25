# Functional Test Suite Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add meaningful functional unit tests covering audio loading, spectrogram parsing, greedy decoding, and the dataset loader, and wire in pytest-cov to enforce a 50% coverage floor in CI.

**Architecture:** Three new test files (audio functional, greedy decoder, spectrogram dataset) hang off a shared `conftest.py` that generates synthetic 16 kHz WAV files in `tmp_path` so no real audio data is required. CI gains a `--cov` flag and a `--cov-fail-under=50` gate on the existing pytest step.

**Tech Stack:** pytest, pytest-cov, soundfile (already a runtime dep), numpy, torch, pandas, scipy

## Global Constraints

- Python ≥ 3.9
- All synthetic audio is a 1-second 440 Hz sine wave at 16 000 Hz sample rate (matches `config.yml sample_rate`)
- `audio_conf` in all tests is a `types.SimpleNamespace`; no YAML loading
- pytest-cov version floor: `>=4.0`
- Coverage flag: `--cov=asr_deepspeech --cov-fail-under=50`
- Do **not** import `asr_deepspeech.models` or `asr_deepspeech.multiproc` (they are in `SKIP_MODULES` for a reason — heavy GPU deps)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Modify | `pyproject.toml` | Add `pytest-cov>=4.0` to `[project.optional-dependencies] test` |
| Modify | `.github/workflows/test.yml` | Append `--cov=asr_deepspeech --cov-fail-under=50` to pytest invocation |
| Create | `tests/conftest.py` | Shared fixtures: synthetic WAV, `audio_conf`, CSV manifest |
| Create | `tests/test_audio_functional.py` | Unit tests for `asr_deepspeech.audio.functional` |
| Create | `tests/test_greedy_decoder.py` | Unit tests for `asr_deepspeech.decoders.GreedyDecoder` |
| Create | `tests/test_spectrogram_dataset.py` | Unit tests for `SpectrogramParser.parse_audio` and `SpectrogramDataset` |

---

### Task 1: Add pytest-cov and CI coverage gate

**Files:**
- Modify: `pyproject.toml` (line ~38, `[project.optional-dependencies]`)
- Modify: `.github/workflows/test.yml` (the `Run tests` step)

**Interfaces:**
- Produces: `pytest-cov` available when running `uv sync --extra test`; CI fails if coverage < 50%

- [ ] **Step 1: Add pytest-cov to pyproject.toml**

  Open `pyproject.toml`. Change the `test` extra from:
  ```toml
  [project.optional-dependencies]
  test = ["pytest"]
  ```
  to:
  ```toml
  [project.optional-dependencies]
  test = ["pytest", "pytest-cov>=4.0"]
  ```

- [ ] **Step 2: Update CI pytest invocation**

  Open `.github/workflows/test.yml`. Change the `Run tests` step from:
  ```yaml
        - name: Run tests
            run: uv run python -m pytest tests/ -v --tb=short
  ```
  to:
  ```yaml
        - name: Run tests
            run: uv run python -m pytest tests/ -v --tb=short --cov=asr_deepspeech --cov-fail-under=50
  ```

- [ ] **Step 3: Verify pyproject.toml parses correctly**

  Run: `uv sync --frozen --extra test`

  Expected: exits 0 with `pytest-cov` resolved (or "Already installed").

- [ ] **Step 4: Commit**

  ```bash
  git add pyproject.toml .github/workflows/test.yml uv.lock
  git commit -m "build: add pytest-cov and 50% coverage gate to CI"
  ```

---

### Task 2: Shared test fixtures (conftest.py)

**Files:**
- Create: `tests/conftest.py`

**Interfaces:**
- Produces:
  - `wav_16k(tmp_path) -> pathlib.Path` — path to a 1-second 16 kHz WAV file
  - `audio_conf() -> types.SimpleNamespace` — standard audio configuration object
  - `manifest_csv(tmp_path, wav_16k) -> pathlib.Path` — path to a two-row CSV manifest

- [ ] **Step 1: Write conftest.py**

  ```python
  import numpy as np
  import pandas as pd
  import pytest
  import soundfile as sf
  from types import SimpleNamespace


  @pytest.fixture
  def wav_16k(tmp_path):
      """1-second 440 Hz sine wave at 16 kHz, saved as a WAV file."""
      sample_rate = 16_000
      t = np.linspace(0, 1, sample_rate, endpoint=False, dtype=np.float32)
      audio = np.sin(2 * np.pi * 440 * t)
      path = tmp_path / "test_audio.wav"
      sf.write(str(path), audio, sample_rate)
      return path


  @pytest.fixture
  def audio_conf():
      return SimpleNamespace(
          sample_rate=16_000,
          window_size=0.02,
          window_stride=0.01,
          window="hamming",
          speed_volume_perturb=False,
          spec_augment=False,
          noise_dir=None,
          noise_prob=0.4,
          noise_levels=(0.0, 0.5),
      )


  @pytest.fixture
  def manifest_csv(tmp_path, wav_16k):
      """Two-row CSV manifest: each row points at the same synthetic WAV with a short label."""
      rows = [
          {"audio_filepath": str(wav_16k), "text": "hello"},
          {"audio_filepath": str(wav_16k), "text": "world"},
      ]
      csv_path = tmp_path / "manifest.csv"
      pd.DataFrame(rows).to_csv(str(csv_path), index=False)
      return csv_path
  ```

- [ ] **Step 2: Verify the fixtures are importable**

  Run: `uv run python -m pytest tests/conftest.py --collect-only`

  Expected: `no tests ran` (fixtures only, 0 errors).

- [ ] **Step 3: Commit**

  ```bash
  git add tests/conftest.py
  git commit -m "test: add shared fixtures (synthetic WAV, audio_conf, CSV manifest)"
  ```

---

### Task 3: Audio functional tests

**Files:**
- Create: `tests/test_audio_functional.py`
- Consumes: `wav_16k` fixture from `tests/conftest.py`

**Interfaces:**
- Consumes:
  - `wav_16k(tmp_path) -> pathlib.Path` (from conftest)
  - `asr_deepspeech.audio.functional.load_audio(path, fq=16000) -> np.ndarray`
  - `asr_deepspeech.audio.functional.duration(path) -> float`
  - `asr_deepspeech.audio.functional.fq(path) -> int`
- Produces: no new interfaces (terminal tests)

- [ ] **Step 1: Write the failing tests**

  ```python
  import numpy as np
  import pytest
  import soundfile as sf

  from asr_deepspeech.audio.functional import duration, fq, load_audio


  def test_load_audio_shape_and_dtype(wav_16k):
      audio = load_audio(str(wav_16k))
      assert audio.ndim == 1
      assert audio.dtype == np.float32
      assert len(audio) == 16_000  # 1-second at 16 kHz


  def test_load_audio_rejects_wrong_sample_rate(wav_16k):
      with pytest.raises(ValueError, match="Expected sample rate"):
          load_audio(str(wav_16k), fq=8_000)


  def test_load_audio_stereo_collapses_to_mono(tmp_path):
      sample_rate = 16_000
      t = np.linspace(0, 1, sample_rate, endpoint=False, dtype=np.float32)
      stereo = np.stack([np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 880 * t)], axis=1)
      path = tmp_path / "stereo.wav"
      sf.write(str(path), stereo, sample_rate)
      audio = load_audio(str(path))
      assert audio.ndim == 1


  def test_duration_is_approximately_one_second(wav_16k):
      d = duration(str(wav_16k))
      assert abs(d - 1.0) < 0.01


  def test_fq_returns_sample_rate(wav_16k):
      assert fq(str(wav_16k)) == 16_000
  ```

- [ ] **Step 2: Run tests to verify they fail before any changes**

  Run: `uv run python -m pytest tests/test_audio_functional.py -v`

  Expected: all tests PASS (the functions already exist; these are functional tests, not TDD for new code). If any FAIL, investigate the error before moving on.

- [ ] **Step 3: Commit**

  ```bash
  git add tests/test_audio_functional.py
  git commit -m "test: add functional tests for audio.functional (load, duration, fq)"
  ```

---

### Task 4: Greedy decoder tests

**Files:**
- Create: `tests/test_greedy_decoder.py`
- Consumes: nothing from conftest (decoder is pure-computation)

**Interfaces:**
- Consumes:
  - `asr_deepspeech.decoders.GreedyDecoder(labels: str, blank_index: int=0)`
  - `GreedyDecoder.process_string(sequence: Tensor, size: int, remove_repetitions: bool) -> (str, Tensor)`
  - `GreedyDecoder.decode(probs: Tensor[batch, seq, classes], sizes=None) -> (list[list[str]], list[list[Tensor]])`
  - `GreedyDecoder.wer(s1: str, s2: str) -> int`
  - `GreedyDecoder.cer(s1: str, s2: str) -> int`
- Produces: no new interfaces (terminal tests)

- [ ] **Step 1: Write the tests**

  ```python
  import torch
  import pytest
  from asr_deepspeech.decoders import GreedyDecoder

  # Labels: blank=0, then A-E, then space
  LABELS = "_ABCDE "


  @pytest.fixture
  def decoder():
      return GreedyDecoder(LABELS, blank_index=0)


  def test_process_string_basic_chars(decoder):
      # indices 1,2,3 → 'A','B','C'
      seq = torch.tensor([1, 2, 3])
      text, offsets = decoder.process_string(seq, 3, remove_repetitions=False)
      assert text == "ABC"
      assert offsets.tolist() == [0, 1, 2]


  def test_process_string_strips_blanks(decoder):
      # blank at position 0 should be dropped; 1,2 → 'A','B'
      seq = torch.tensor([0, 1, 2])
      text, offsets = decoder.process_string(seq, 3, remove_repetitions=False)
      assert text == "AB"
      assert offsets.tolist() == [1, 2]


  def test_process_string_removes_repetitions(decoder):
      # [1, 1, 2] with remove_repetitions=True → 'A','B' (second A is skipped)
      seq = torch.tensor([1, 1, 2])
      text, offsets = decoder.process_string(seq, 3, remove_repetitions=True)
      assert text == "AB"


  def test_process_string_keeps_repetitions_when_disabled(decoder):
      # [1, 1, 2] with remove_repetitions=False → 'A','A','B'
      seq = torch.tensor([1, 1, 2])
      text, offsets = decoder.process_string(seq, 3, remove_repetitions=False)
      assert text == "AAB"


  def test_process_string_space_char(decoder):
      # index 6 is space in LABELS
      seq = torch.tensor([1, 6, 2])
      text, offsets = decoder.process_string(seq, 3, remove_repetitions=False)
      assert text == "A B"


  def test_decode_argmax_path(decoder):
      # probs shape: (1, 3, 7); argmax → [1, 2, 3] → "ABC"
      probs = torch.zeros(1, 3, len(LABELS))
      probs[0, 0, 1] = 1.0  # A
      probs[0, 1, 2] = 1.0  # B
      probs[0, 2, 3] = 1.0  # C
      strings, offsets = decoder.decode(probs)
      assert strings[0][0] == "ABC"


  def test_decode_removes_consecutive_repeats(decoder):
      # argmax → [1, 1, 2]; decode collapses to "AB"
      probs = torch.zeros(1, 3, len(LABELS))
      probs[0, 0, 1] = 1.0  # A
      probs[0, 1, 1] = 1.0  # A (repeat)
      probs[0, 2, 2] = 1.0  # B
      strings, _ = decoder.decode(probs)
      assert strings[0][0] == "AB"


  def test_decode_removes_blank_tokens(decoder):
      # argmax → [0, 1, 2]; blank at position 0 is stripped → "AB"
      probs = torch.zeros(1, 3, len(LABELS))
      probs[0, 0, 0] = 1.0  # blank
      probs[0, 1, 1] = 1.0  # A
      probs[0, 2, 2] = 1.0  # B
      strings, _ = decoder.decode(probs)
      assert strings[0][0] == "AB"


  def test_decode_batch_of_two(decoder):
      probs = torch.zeros(2, 2, len(LABELS))
      probs[0, 0, 1] = 1.0  # A
      probs[0, 1, 2] = 1.0  # B
      probs[1, 0, 3] = 1.0  # C
      probs[1, 1, 4] = 1.0  # D
      strings, _ = decoder.decode(probs)
      assert strings[0][0] == "AB"
      assert strings[1][0] == "CD"


  def test_wer_identical_sentences(decoder):
      assert decoder.wer("hello world", "hello world") == 0


  def test_wer_one_substitution(decoder):
      assert decoder.wer("hello world", "hello earth") == 1


  def test_cer_identical_strings(decoder):
      assert decoder.cer("hello", "hello") == 0


  def test_cer_one_char_difference(decoder):
      assert decoder.cer("hello", "helo") == 1
  ```

- [ ] **Step 2: Run the tests**

  Run: `uv run python -m pytest tests/test_greedy_decoder.py -v`

  Expected: all PASS. If any fail, inspect the failure message — the most likely cause is an off-by-one in `process_string` or `int_to_char` mapping.

- [ ] **Step 3: Commit**

  ```bash
  git add tests/test_greedy_decoder.py
  git commit -m "test: add unit tests for GreedyDecoder (process_string, decode, wer, cer)"
  ```

---

### Task 5: SpectrogramParser and SpectrogramDataset tests

**Files:**
- Create: `tests/test_spectrogram_dataset.py`
- Consumes: `wav_16k`, `audio_conf`, `manifest_csv` fixtures from `tests/conftest.py`

**Interfaces:**
- Consumes:
  - `wav_16k`, `audio_conf`, `manifest_csv` from conftest
  - `asr_deepspeech.data.parsers.SpectrogramParser(audio_conf, normalize, speed_volume_perturb, spec_augment)`
  - `SpectrogramParser.parse_audio(audio_path: str) -> torch.FloatTensor`  shape `(freq_bins, time_frames)`
  - `asr_deepspeech.data.dataset.SpectrogramDataset(audio_conf, manifest_filepath, labels, normalize, spec_augment, caching)`
  - `SpectrogramDataset.__len__() -> int`
  - `SpectrogramDataset.__getitem__(index) -> (torch.FloatTensor, list[int])`
  - `SpectrogramDataset.parse_transcript(text: str) -> list[int]`
- Produces: no new interfaces (terminal tests)

- [ ] **Step 1: Write the tests**

  ```python
  import torch
  import pytest
  from asr_deepspeech.data.parsers import SpectrogramParser
  from asr_deepspeech.data.dataset import SpectrogramDataset

  # Labels map: char -> int index.  Index 0 is intentionally unused (CTC blank).
  LABELS_MAP = {c: i + 1 for i, c in enumerate("abcdefghijklmnopqrstuvwxyz ")}
  # Expected STFT freq bins: n_fft/2 + 1 = int(16000 * 0.02)/2 + 1 = 161
  EXPECTED_FREQ_BINS = 161


  # ── SpectrogramParser ────────────────────────────────────────────────────────


  def test_parse_audio_returns_float_tensor(wav_16k, audio_conf):
      parser = SpectrogramParser(audio_conf, normalize=False)
      spect = parser.parse_audio(str(wav_16k))
      assert isinstance(spect, torch.FloatTensor)


  def test_parse_audio_freq_bins(wav_16k, audio_conf):
      parser = SpectrogramParser(audio_conf, normalize=False)
      spect = parser.parse_audio(str(wav_16k))
      assert spect.shape[0] == EXPECTED_FREQ_BINS


  def test_parse_audio_time_frames_positive(wav_16k, audio_conf):
      parser = SpectrogramParser(audio_conf, normalize=False)
      spect = parser.parse_audio(str(wav_16k))
      assert spect.shape[1] > 0


  def test_parse_audio_normalize_zero_mean(wav_16k, audio_conf):
      parser = SpectrogramParser(audio_conf, normalize=True)
      spect = parser.parse_audio(str(wav_16k))
      assert abs(spect.mean().item()) < 0.1


  def test_parse_audio_all_values_finite(wav_16k, audio_conf):
      parser = SpectrogramParser(audio_conf, normalize=False)
      spect = parser.parse_audio(str(wav_16k))
      assert torch.isfinite(spect).all()


  # ── SpectrogramDataset ───────────────────────────────────────────────────────


  def test_dataset_len(manifest_csv, audio_conf):
      ds = SpectrogramDataset(
          audio_conf=audio_conf,
          manifest_filepath=str(manifest_csv),
          labels=LABELS_MAP,
          normalize=False,
      )
      assert len(ds) == 2


  def test_dataset_getitem_returns_tensor_and_list(manifest_csv, audio_conf):
      ds = SpectrogramDataset(
          audio_conf=audio_conf,
          manifest_filepath=str(manifest_csv),
          labels=LABELS_MAP,
          normalize=False,
      )
      spect, transcript = ds[0]
      assert isinstance(spect, torch.FloatTensor)
      assert isinstance(transcript, list)


  def test_dataset_getitem_spectrogram_shape(manifest_csv, audio_conf):
      ds = SpectrogramDataset(
          audio_conf=audio_conf,
          manifest_filepath=str(manifest_csv),
          labels=LABELS_MAP,
          normalize=False,
      )
      spect, _ = ds[0]
      assert spect.shape[0] == EXPECTED_FREQ_BINS
      assert spect.shape[1] > 0


  def test_dataset_parse_transcript_known_chars(manifest_csv, audio_conf):
      ds = SpectrogramDataset(
          audio_conf=audio_conf,
          manifest_filepath=str(manifest_csv),
          labels=LABELS_MAP,
          normalize=False,
      )
      # "hello" → [h=8, e=5, l=12, l=12, o=15]
      result = ds.parse_transcript("hello")
      assert result == [
          LABELS_MAP["h"],
          LABELS_MAP["e"],
          LABELS_MAP["l"],
          LABELS_MAP["l"],
          LABELS_MAP["o"],
      ]


  def test_dataset_parse_transcript_drops_unknown_chars(manifest_csv, audio_conf):
      ds = SpectrogramDataset(
          audio_conf=audio_conf,
          manifest_filepath=str(manifest_csv),
          labels=LABELS_MAP,
          normalize=False,
      )
      # digits are not in LABELS_MAP → filtered out
      result = ds.parse_transcript("a1b")
      assert result == [LABELS_MAP["a"], LABELS_MAP["b"]]


  def test_dataset_both_items_accessible(manifest_csv, audio_conf):
      ds = SpectrogramDataset(
          audio_conf=audio_conf,
          manifest_filepath=str(manifest_csv),
          labels=LABELS_MAP,
          normalize=False,
      )
      for i in range(len(ds)):
          spect, transcript = ds[i]
          assert spect.shape[0] == EXPECTED_FREQ_BINS
          assert len(transcript) > 0
  ```

- [ ] **Step 2: Run the tests**

  Run: `uv run python -m pytest tests/test_spectrogram_dataset.py -v`

  Expected: all PASS. If `test_parse_audio_normalize_zero_mean` fails with a larger mean, check that `normalize=True` is properly calling `spect.add_(-mean)` in `SpectrogramParser.parse_audio`.

- [ ] **Step 3: Run full suite with coverage**

  Run: `uv run python -m pytest tests/ -v --tb=short --cov=asr_deepspeech --cov-fail-under=50`

  Expected: all tests pass and coverage ≥ 50%. If coverage is below threshold, add tests or inspect the report to find untested hot paths.

- [ ] **Step 4: Commit**

  ```bash
  git add tests/test_spectrogram_dataset.py
  git commit -m "test: add unit tests for SpectrogramParser and SpectrogramDataset"
  ```

---

## Self-Review

**Spec coverage check:**

| Requirement | Covered by |
|-------------|-----------|
| Add pytest-cov to test deps | Task 1, `pyproject.toml` change |
| `--cov=asr_deepspeech --cov-fail-under=50` in CI | Task 1, `.github/workflows/test.yml` change |
| Unit test: audio functional (windowing) | Task 4 — `parse_audio` exercises STFT windowing |
| Unit test: greedy decoder | Task 4 — `test_greedy_decoder.py` |
| Unit test: data loader with synthetic audio fixture | Task 5 — `SpectrogramDataset` + `conftest.py` `wav_16k` fixture |

**Placeholder scan:** None found — every step contains complete code.

**Type consistency check:**
- `wav_16k` fixture returns `pathlib.Path`; callers cast with `str(wav_16k)` ✓
- `audio_conf` is `SimpleNamespace`; `SpectrogramParser` accesses `.window_stride`, `.window_size`, `.sample_rate`, `.window`, `.noise_dir`, `.noise_prob` — all present in fixture ✓
- `SpectrogramDataset` receives `labels=LABELS_MAP` (a `dict`); the `if type(labels) == str` branch in `__init__` correctly falls through to `self.labels_map = labels` ✓
- `LABELS_MAP` values start at 1 (not 0), sidestepping the `filter(None, ...)` silent-drop-of-zero-index issue ✓
