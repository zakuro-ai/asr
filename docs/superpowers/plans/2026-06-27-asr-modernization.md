# asr_deepspeech Modernization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make asr_deepspeech device-agnostic (CPU/GPU auto), remove deprecated torch APIs, delete dead code + visdom, remove `eval()`, add core type hints, and expand tests — with GPU numerics unchanged.

**Architecture:** A new `asr_deepspeech/device.py` centralizes device + AMP resolution. Callers (config, trainer, model inference) route through it. Deprecated tensor/dist constructors are replaced with modern equivalents. Dead modules are deleted. Each change is covered by a focused test.

**Tech Stack:** Python 3.9+ (CI runs 3.10 & 3.12), PyTorch ≥2.0 (env has 2.12), uv, pytest + pytest-cov, ruff.

## Global Constraints

- **Python floor:** `requires-python = ">=3.9"`; `ruff target-version = py39`. No 3.10+ only syntax.
- **Lint:** `ruff check .` and `ruff format --check .` must pass (line-length 100, rules `E,F,W,I`).
- **Tests:** `uv run python -m pytest tests/ --cov=asr_deepspeech --cov-fail-under=50` must pass (gate ≥50%).
- **No modeling changes:** GPU runs must stay bit-for-bit identical. AMP enabled only on CUDA.
- **Backward compatible config:** existing `device: cuda`, `rnn_type: nn.LSTM`, `betas: "(0.9, 0.999)"` values must keep working.
- **Commit identity:** author `CADIC Jean Maximilien <jmcadic.me@gmail.com>`; every commit body ends with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.
- **Work in the isolated worktree:** `/opt/code/ZAK/asr-upgrade` on branch `feat/modernize-device-portability`. Use `uv run` for all python.

---

### Task 1: Device + AMP resolution helper

**Files:**
- Create: `asr_deepspeech/device.py`
- Test: `tests/test_device.py`

**Interfaces:**
- Produces:
  - `resolve_device(spec="auto") -> torch.device` — accepts `"auto"|"cuda"|"gpu"|"cpu"`, a `torch.device`, or `None` (→auto); `"auto"`→cuda if available else cpu; explicit `"cuda"` with no GPU → warns + cpu; unknown → `ValueError`.
  - `make_grad_scaler(device, enabled=True) -> torch.amp.GradScaler` — enabled only when resolved `device.type=="cuda"` and `enabled`.
  - `autocast(device, enabled=True)` — `torch.amp.autocast` context; no-op unless cuda + enabled.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_device.py
import pytest
import torch

from asr_deepspeech.device import autocast, make_grad_scaler, resolve_device


def test_resolve_cpu_explicit():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_auto_is_cpu_without_cuda():
    if not torch.cuda.is_available():
        assert resolve_device("auto").type == "cpu"


def test_resolve_none_defaults_auto():
    assert isinstance(resolve_device(None), torch.device)


def test_resolve_passthrough_device_object():
    d = torch.device("cpu")
    assert resolve_device(d) == d


def test_resolve_cuda_without_gpu_warns_and_falls_back():
    if not torch.cuda.is_available():
        with pytest.warns(UserWarning):
            assert resolve_device("cuda").type == "cpu"


def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        resolve_device("tpu")


def test_grad_scaler_disabled_on_cpu():
    scaler = make_grad_scaler("cpu", enabled=True)
    assert scaler.is_enabled() is False


def test_autocast_cpu_is_noop_context():
    with autocast("cpu", enabled=True):
        x = torch.ones(2) + 1
    assert x.dtype == torch.float32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_device.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'asr_deepspeech.device'`

- [ ] **Step 3: Write minimal implementation**

```python
# asr_deepspeech/device.py
"""Device + mixed-precision resolution, centralized for CPU/GPU portability."""

import warnings

import torch


def resolve_device(spec="auto") -> torch.device:
    """Resolve a device spec to a concrete ``torch.device``.

    Accepts ``"auto"``, ``"cuda"``/``"gpu"``, ``"cpu"``, a ``torch.device``,
    or ``None`` (treated as ``"auto"``). ``"auto"`` selects CUDA when available,
    otherwise CPU. An explicit CUDA request on a machine without a GPU warns and
    falls back to CPU. Unknown specs raise ``ValueError``.
    """
    if spec is None:
        spec = "auto"
    if isinstance(spec, torch.device):
        return spec
    key = str(spec).lower()
    if key == "cpu":
        return torch.device("cpu")
    if key in ("cuda", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        warnings.warn(
            "CUDA device requested but no GPU is available; falling back to CPU.",
            UserWarning,
            stacklevel=2,
        )
        return torch.device("cpu")
    if key == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unknown device spec: {spec!r}")


def make_grad_scaler(device, enabled: bool = True) -> "torch.amp.GradScaler":
    """GradScaler that is enabled only for CUDA (AMP is a no-op elsewhere)."""
    dev = resolve_device(device)
    return torch.amp.GradScaler(dev.type, enabled=enabled and dev.type == "cuda")


def autocast(device, enabled: bool = True):
    """Autocast context; a no-op unless running on CUDA with ``enabled``."""
    dev = resolve_device(device)
    return torch.amp.autocast(device_type=dev.type, enabled=enabled and dev.type == "cuda")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run python -m pytest tests/test_device.py -v`
Expected: PASS (8 passed)

- [ ] **Step 5: Lint + commit**

```bash
cd /opt/code/ZAK/asr-upgrade
uv tool run ruff check asr_deepspeech/device.py tests/test_device.py
uv tool run ruff format asr_deepspeech/device.py tests/test_device.py
git add asr_deepspeech/device.py tests/test_device.py
git -c user.name="CADIC Jean Maximilien" -c user.email="jmcadic.me@gmail.com" \
  commit --author="CADIC Jean Maximilien <jmcadic.me@gmail.com>" \
  -m "feat(device): add resolve_device + AMP helpers for CPU/GPU portability

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Wire device portability into config, vars, model inference, and trainer

**Files:**
- Modify: `asr_deepspeech/vars.py:13-14` (guard cuda seed)
- Modify: `asr_deepspeech/config.yml:29-30` (`device`/`device_test` → `auto`)
- Modify: `asr_deepspeech/modules/deepspeech.py:157-181` (`__call__` device default + resolve)
- Modify: `asr_deepspeech/trainers/deepspeech_trainer.py` (resolve device in `__init__`; device-agnostic AMP in `train`)
- Test: `tests/test_device_wiring.py`

**Interfaces:**
- Consumes: `resolve_device`, `make_grad_scaler`, `autocast` from Task 1.
- Produces: `DeepSpeech.__call__(device="cpu", ...)` resolves any spec; `DeepSpeechTrainer` stores a concrete `torch.device`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_device_wiring.py
import torch

import asr_deepspeech.vars as v  # must import without crashing on CPU


def test_vars_imports_on_cpu():
    assert v.supported_rnns["lstm"] is torch.nn.LSTM


def test_deepspeech_call_defaults_to_cpu(audio_conf, label_csv):
    from asr_deepspeech.modules.deepspeech import DeepSpeech

    model = DeepSpeech(
        audio_conf=audio_conf, decoder=None, label_path=str(label_csv),
        rnn_type="nn.GRU", rnn_hidden_size=16, rnn_hidden_layers=1, bidirectional=True,
    )
    model.eval()
    inputs = torch.randn(1, 1, 161, 40)
    loader = [(inputs, torch.tensor([1, 2, 3], dtype=torch.int32),
               torch.tensor([1.0]), torch.tensor([3], dtype=torch.int32))]
    wer, cer, _ = model(loader=loader)  # no device arg -> cpu, must not raise
    assert isinstance(wer, float)
```

(The `audio_conf` fixture lives in `tests/conftest.py`; add a shared `label_csv`
fixture there — see Step 3 — so multiple test modules can use it.)

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_device_wiring.py -v`
Expected: FAIL — `model(...)` defaults `device="cuda"` → `RuntimeError: Found no NVIDIA driver` (and/or `label_csv` fixture missing).

- [ ] **Step 3: Apply the wiring changes**

`asr_deepspeech/vars.py` — guard the CUDA seed:

```python
torch.manual_seed(123456)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123456)
```

`asr_deepspeech/config.yml` — lines 29-30:

```yaml
  device:                   auto
  device_test:              auto
```

`asr_deepspeech/modules/deepspeech.py` — add import near the top (after `import torch`):

```python
from asr_deepspeech.device import resolve_device
```

Change the `__call__` signature default and resolve at the top of the body:

```python
    def __call__(
        self,
        loader=None,
        manifest=None,
        batch_size=None,
        device="cpu",
        num_workers=32,
        dist=None,
        verbose=False,
        half=False,
        output_file=None,
        main_proc=True,
        restart_from=None,
        cuda=True,
    ):
        device = resolve_device(device)
        with torch.no_grad():
```

`asr_deepspeech/trainers/deepspeech_trainer.py` — add imports:

```python
from asr_deepspeech.device import autocast, make_grad_scaler, resolve_device
```

In `__init__`, resolve the two device args before `super().__init__`:

```python
        device = resolve_device(device)
        device_test = resolve_device(device_test)
        super(DeepSpeechTrainer, self).__init__(
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=metrics,
            epochs=epochs,
            model_path=model_path,
            checkpoint_path=checkpoint_path,
            device=device,
            device_test=device_test,
        )
```

Replace the AMP usage in `train()` (lines ~69-88) with device-agnostic, CUDA-gated AMP:

```python
        use_amp = self.mixed_precision and self._device.type == "cuda"
        scaler = make_grad_scaler(self._device, enabled=self.mixed_precision)

        for iter, data in tqdm(
            enumerate(loader, start=0), total=len(loader), desc=self.description()
        ):
            if use_amp:
                with autocast(self._device, enabled=True):
                    valid_loss, loss, loss_value = self.fit(data)
            else:
                valid_loss, loss, loss_value = self.fit(data)

            if valid_loss:
                self._optimizer.zero_grad()
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(self._optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    self._optimizer.step()
                current.loss += loss_value
            else:
                print("Loss non valid, skipped")
```

Add the shared fixture to `tests/conftest.py`:

```python
@pytest.fixture
def label_csv(tmp_path):
    """Minimal label CSV with 26 characters (a-z)."""
    chars = list("abcdefghijklmnopqrstuvwxyz")
    path = tmp_path / "labels.csv"
    pd.DataFrame({"label": chars}).to_csv(str(path), index=False)
    return path
```

(There is an identically-named `label_csv` fixture defined locally inside
`tests/test_spectrogram_dataset.py`; once it lives in `conftest.py`, delete the
local copy there to avoid the duplicate-fixture shadow.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_device_wiring.py tests/test_spectrogram_dataset.py -v`
Expected: PASS (new wiring tests + existing model tests still green)

- [ ] **Step 5: Lint + commit**

```bash
cd /opt/code/ZAK/asr-upgrade
uv tool run ruff check . && uv tool run ruff format --check .
git add asr_deepspeech/vars.py asr_deepspeech/config.yml asr_deepspeech/modules/deepspeech.py \
        asr_deepspeech/trainers/deepspeech_trainer.py tests/conftest.py tests/test_spectrogram_dataset.py \
        tests/test_device_wiring.py
git -c user.name="CADIC Jean Maximilien" -c user.email="jmcadic.me@gmail.com" \
  commit --author="CADIC Jean Maximilien <jmcadic.me@gmail.com>" \
  -m "feat(device): auto-resolve device everywhere; CUDA-gated AMP

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Remove `eval()` — rnn_type lookup and betas parsing

**Files:**
- Modify: `asr_deepspeech/vars.py` (add `resolve_rnn_type`)
- Modify: `asr_deepspeech/modules/deepspeech.py:43` (use lookup, not `eval`)
- Modify: `asr_deepspeech/trainers/__main__.py:34` (parse betas with `ast.literal_eval`)
- Test: `tests/test_rnn_type.py`

**Interfaces:**
- Consumes: `supported_rnns` (already in `vars.py`).
- Produces: `resolve_rnn_type(spec) -> type[nn.RNNBase]` accepting `"nn.LSTM"`, `"LSTM"`, `"lstm"`, or an `nn.Module` subclass (passthrough).
- Produces: `parse_betas(value) -> tuple[float, float]` in `trainers/__main__.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_rnn_type.py
import torch.nn as nn

from asr_deepspeech.vars import resolve_rnn_type


def test_resolve_dotted_string():
    assert resolve_rnn_type("nn.LSTM") is nn.LSTM


def test_resolve_bare_lowercase():
    assert resolve_rnn_type("gru") is nn.GRU


def test_resolve_uppercase():
    assert resolve_rnn_type("RNN") is nn.RNN


def test_resolve_passthrough_class():
    assert resolve_rnn_type(nn.LSTM) is nn.LSTM
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_rnn_type.py -v`
Expected: FAIL — `ImportError: cannot import name 'resolve_rnn_type'`

- [ ] **Step 3: Implement**

Append to `asr_deepspeech/vars.py`:

```python
def resolve_rnn_type(spec):
    """Map an RNN spec to its class without ``eval``.

    Accepts ``"nn.LSTM"``, ``"LSTM"``, ``"lstm"`` (and rnn/gru variants) or an
    already-resolved class (passthrough).
    """
    if isinstance(spec, type):
        return spec
    key = str(spec).split(".")[-1].lower()
    try:
        return supported_rnns[key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported rnn_type {spec!r}; expected one of {sorted(supported_rnns)}"
        ) from exc
```

`asr_deepspeech/modules/deepspeech.py` — replace line 43:

```python
        self.rnn_type = resolve_rnn_type(rnn_type)
```

and add the import near the other `asr_deepspeech` imports:

```python
from asr_deepspeech.vars import resolve_rnn_type
```

`asr_deepspeech/trainers/__main__.py` — add at top:

```python
import ast


def parse_betas(value):
    """Parse optimizer betas from a string like "(0.9, 0.999)" or a native list."""
    if isinstance(value, (tuple, list)):
        return tuple(float(x) for x in value)
    return tuple(float(x) for x in ast.literal_eval(value))
```

and replace `betas=eval(cfg.optim.betas)` with `betas=parse_betas(cfg.optim.betas)`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_rnn_type.py tests/test_spectrogram_dataset.py -v`
Expected: PASS (model still builds with `"nn.GRU"` string in existing tests)

- [ ] **Step 5: Lint + commit**

```bash
cd /opt/code/ZAK/asr-upgrade
uv tool run ruff check . && uv tool run ruff format --check .
git add asr_deepspeech/vars.py asr_deepspeech/modules/deepspeech.py \
        asr_deepspeech/trainers/__main__.py tests/test_rnn_type.py
git -c user.name="CADIC Jean Maximilien" -c user.email="jmcadic.me@gmail.com" \
  commit --author="CADIC Jean Maximilien <jmcadic.me@gmail.com>" \
  -m "refactor: replace eval() with rnn_type lookup and safe betas parsing

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Replace deprecated tensor/dist APIs

**Files:**
- Modify: `asr_deepspeech/functional.py:19-20,31,38`
- Modify: `asr_deepspeech/data/parsers/spectrogram_parser.py:55`
- Modify: `asr_deepspeech/modules/blocks.py:50-52`
- Test: `tests/test_blocks_mask.py`, extend `tests/test_audio_functional.py`

**Interfaces:**
- No new public API. Behavior is preserved; only constructors/enums change.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_blocks_mask.py
import torch

from asr_deepspeech.modules.blocks import MaskConv


def test_maskconv_runs_on_cpu_and_masks_padding():
    seq = torch.nn.Sequential(torch.nn.Conv2d(1, 1, kernel_size=3, padding=1))
    mod = MaskConv(seq)
    x = torch.randn(2, 1, 8, 10)
    lengths = torch.tensor([10, 4])
    out, out_lengths = mod(x, lengths)
    assert out.shape == x.shape
    # Second sample: everything past length 4 in the time dim must be zeroed.
    assert torch.count_nonzero(out[1, 0, :, 4:]) == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run python -m pytest tests/test_blocks_mask.py -v`
Expected: PASS today **iff** `torch.BoolTensor` still works — run it; if it passes, this is a characterization test that must stay green after the refactor. (If it errors on the deprecated constructor, it fails now.) Either way it guards the behavior.

- [ ] **Step 3: Implement the replacements**

`asr_deepspeech/modules/blocks.py` lines 50-52 → device-aware construction:

```python
            mask = torch.zeros(x.size(), dtype=torch.bool, device=x.device)
```

(delete the following `if x.is_cuda: mask = mask.cuda()` lines — `device=x.device` handles it.)

`asr_deepspeech/functional.py` — replace the deprecated constructors and enum:

```python
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.zeros(minibatch_size, dtype=torch.float32)
    target_sizes = torch.zeros(minibatch_size, dtype=torch.int32)
```

```python
    targets = torch.tensor(targets, dtype=torch.int32)
```

```python
    dist.all_reduce(
        rt, op=dist.ReduceOp.MAX if reduce_op_max is True else dist.ReduceOp.SUM
    )  # Default to sum
```

`asr_deepspeech/data/parsers/spectrogram_parser.py:55`:

```python
        spect = torch.from_numpy(spect).float()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run python -m pytest tests/test_blocks_mask.py tests/test_audio_functional.py tests/test_spectrogram_dataset.py -v`
Expected: PASS (mask behavior + collate + parser shapes unchanged)

- [ ] **Step 5: Lint + commit**

```bash
cd /opt/code/ZAK/asr-upgrade
uv tool run ruff check . && uv tool run ruff format --check .
git add asr_deepspeech/functional.py asr_deepspeech/data/parsers/spectrogram_parser.py \
        asr_deepspeech/modules/blocks.py tests/test_blocks_mask.py
git -c user.name="CADIC Jean Maximilien" -c user.email="jmcadic.me@gmail.com" \
  commit --author="CADIC Jean Maximilien <jmcadic.me@gmail.com>" \
  -m "refactor: replace deprecated FloatTensor/IntTensor/BoolTensor + dist.reduce_op

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Remove dead code, visdom, and the six dependency

**Files:**
- Delete: `asr_deepspeech/models/deepspeech_model.py`
- Modify/Delete: `asr_deepspeech/models/__init__.py`
- Delete: `asr_deepspeech/loggers/visdom_logger.py`
- Modify: `asr_deepspeech/parsers/default_parser.py` (drop `--visdom`; fix `--id` help)
- Modify: `asr_deepspeech/decoders/greedy_decoder.py:2,17` (`six.moves.xrange` → `range`)
- Modify: `tests/test_imports.py` (drop the skipped dead module)

**Interfaces:**
- Removes `asr_deepspeech.models.DeepSpeechModel` and `VisdomLogger` (neither is imported at runtime — verified by grep).

- [ ] **Step 1: Confirm nothing imports the dead symbols**

Run:
```bash
cd /opt/code/ZAK/asr-upgrade
grep -rn "DeepSpeechModel\|VisdomLogger\|six" asr_deepspeech/ tests/ | grep -v "deepspeech_model.py\|visdom_logger.py"
```
Expected: only references in `models/__init__.py`, `default_parser.py` (`--visdom`, help text), `greedy_decoder.py` (`six.moves`), and `tests/test_imports.py`. No runtime importers.

- [ ] **Step 2: Delete dead files and edit references**

```bash
cd /opt/code/ZAK/asr-upgrade
git rm asr_deepspeech/models/deepspeech_model.py asr_deepspeech/loggers/visdom_logger.py
```

`asr_deepspeech/models/__init__.py` — replace its single import line with:

```python
# DeepSpeechModel was removed (dead apex/visdom code). The model lives in
# asr_deepspeech.modules.deepspeech.DeepSpeech.
```

`asr_deepspeech/decoders/greedy_decoder.py` — delete line 2 (`from six.moves import xrange`) and change line 17 `for x in xrange(len(sequences)):` → `for x in range(len(sequences)):`.

`asr_deepspeech/parsers/default_parser.py` — delete the `--visdom` argument line; change the `--id` help text from `"Identifier for visdom/tensorboard run"` to `"Identifier for the tensorboard run"`.

`tests/test_imports.py` — remove `"asr_deepspeech.models.deepspeech_model"` from the skip list (and, if the list becomes empty, simplify the test to assert the package imports).

- [ ] **Step 3: Run the import + decoder tests**

Run: `uv run python -m pytest tests/test_imports.py tests/test_greedy_decoder.py -v`
Expected: PASS

- [ ] **Step 4: Verify nothing else broke**

Run: `uv run python -c "import asr_deepspeech.models, asr_deepspeech.loggers, asr_deepspeech.decoders; print('ok')"`
Expected: prints `ok`

- [ ] **Step 5: Lint + commit**

```bash
cd /opt/code/ZAK/asr-upgrade
uv tool run ruff check . && uv tool run ruff format --check .
git add -A
git -c user.name="CADIC Jean Maximilien" -c user.email="jmcadic.me@gmail.com" \
  commit --author="CADIC Jean Maximilien <jmcadic.me@gmail.com>" \
  -m "chore: remove dead DeepSpeechModel, visdom logger, and six dependency

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Type hints on core methods

**Files:**
- Modify: `asr_deepspeech/modules/deepspeech.py` (`forward`, `get_seq_lens`)
- Modify: `asr_deepspeech/trainers/deepspeech_trainer.py` (`fit`)
- Modify: `asr_deepspeech/device.py` (already hinted in Task 1 — verify)

**Interfaces:** No behavior change. Hints only.

- [ ] **Step 1: Add hints to `DeepSpeech.forward` and `get_seq_lens`**

```python
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
```

```python
    def get_seq_lens(self, input_length: torch.Tensor) -> torch.Tensor:
```

(Use `from __future__ import annotations` at the top of any file using `tuple[...]`
builtin generics, since `target-version = py39`. Add it as the first line if absent.)

- [ ] **Step 2: Add hints to `DeepSpeechTrainer.fit`**

```python
    def fit(self, data) -> tuple[bool, torch.Tensor, float]:
```

(Add `from __future__ import annotations` at the top of `deepspeech_trainer.py`.)

- [ ] **Step 3: Run the full suite + lint**

Run:
```bash
cd /opt/code/ZAK/asr-upgrade
uv tool run ruff check . && uv tool run ruff format --check .
uv run python -m pytest tests/ -q
```
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add asr_deepspeech/modules/deepspeech.py asr_deepspeech/trainers/deepspeech_trainer.py
git -c user.name="CADIC Jean Maximilien" -c user.email="jmcadic.me@gmail.com" \
  commit --author="CADIC Jean Maximilien <jmcadic.me@gmail.com>" \
  -m "refactor: add type hints to core model + trainer methods

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: CTC training-step test, full verification, and PR

**Files:**
- Create: `tests/test_training_step.py`

**Interfaces:**
- Consumes: `DeepSpeech` (model), `resolve_device`.

- [ ] **Step 1: Write the training-step test (real CTC loss decreases on CPU)**

```python
# tests/test_training_step.py
import torch
import torch.nn as nn

from asr_deepspeech.modules.deepspeech import DeepSpeech

FREQ_BINS = 161


def _model(label_csv, audio_conf):
    return DeepSpeech(
        audio_conf=audio_conf, decoder=None, label_path=str(label_csv),
        rnn_type="nn.GRU", rnn_hidden_size=32, rnn_hidden_layers=1, bidirectional=True,
    )


def test_ctc_train_step_loss_decreases_on_cpu(label_csv, audio_conf):
    torch.manual_seed(0)
    model = _model(label_csv, audio_conf).to("cpu")
    model.train()

    inputs = torch.randn(2, 1, FREQ_BINS, 40)
    targets = torch.randint(1, 26, (2, 3), dtype=torch.int32)
    flat_targets = targets.reshape(-1)
    target_sizes = torch.tensor([3, 3], dtype=torch.int32)
    input_percentages = torch.ones(2)

    criterion = nn.CTCLoss(blank=0, reduction="sum", zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    def step():
        input_sizes = input_percentages.mul(int(inputs.size(3))).int()
        out, output_sizes = model.forward(inputs, input_sizes)
        out = out.transpose(0, 1).float().log_softmax(2)
        return criterion(out, flat_targets, output_sizes, target_sizes) / inputs.size(0)

    first = None
    for i in range(40):
        opt.zero_grad()
        loss = step()
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
    assert loss.item() < first  # the model is learning
```

- [ ] **Step 2: Run it**

Run: `uv run python -m pytest tests/test_training_step.py -v`
Expected: PASS (loss after 40 steps < initial loss)

- [ ] **Step 3: Full CI-equivalent gate**

Run:
```bash
cd /opt/code/ZAK/asr-upgrade
uv tool run ruff check .
uv tool run ruff format --check .
uv run python -m pytest tests/ --cov=asr_deepspeech --cov-report=term-missing --cov-fail-under=50
```
Expected: ruff clean; all tests pass; total coverage ≥50%.

- [ ] **Step 4: Run the end-to-end training cycle harness as evidence**

Run: `uv run python /tmp/claude-1000/-home-foo/0aec53b1-3c7e-417a-8c05-b28129bc5441/scratchpad/train_cycle.py`
Expected: Part A now shows `resolve_device("auto")` → cpu (no crash); Part B loss decreases; decode runs.

- [ ] **Step 5: Commit, push, open PR**

```bash
cd /opt/code/ZAK/asr-upgrade
git add tests/test_training_step.py
git -c user.name="CADIC Jean Maximilien" -c user.email="jmcadic.me@gmail.com" \
  commit --author="CADIC Jean Maximilien <jmcadic.me@gmail.com>" \
  -m "test: add CPU CTC training-step regression (loss decreases)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
git push -u origin feat/modernize-device-portability
gh pr create --repo zakuro-ai/asr --base master --head feat/modernize-device-portability \
  --title "feat: modernize asr — device portability, deprecation fixes, dead-code removal" \
  --body "See docs/superpowers/specs/2026-06-27-asr-modernization-design.md. CPU/GPU auto, torch API modernization, dead-code + visdom + six removal, eval() removal, core type hints, expanded tests. GPU numerics unchanged."
```

- [ ] **Step 6: Wait for CI green, then report** (do not auto-merge without the user's go-ahead for this feature PR)

---

## Self-Review

**Spec coverage:**
- Unit 1 (device portability) → Tasks 1, 2 ✓
- Unit 2 (device-agnostic AMP) → Tasks 1 (helpers), 2 (trainer wiring) ✓
- Unit 3 (deprecated tensor/dist APIs) → Task 4 ✓
- Unit 4 (remove eval) → Task 3 ✓
- Unit 5 (dead code + visdom + six) → Task 5 ✓
- Unit 6 (type hints) → Task 6 ✓
- Unit 7 (tests + packaging tidy) → Tasks 1-7 tests; coverage gate in Task 7; six removed in Task 5 ✓

**Placeholder scan:** No TBD/TODO; every code step shows actual code. ✓

**Type consistency:** `resolve_device` / `make_grad_scaler` / `autocast` signatures used consistently across Tasks 1, 2, 7. `resolve_rnn_type` defined in Task 3 and used in `deepspeech.py`. `parse_betas` defined and used in `trainers/__main__.py`. ✓

**Note for executor:** `pyproject.toml` does not list `six` as a dependency (it was an undeclared transitive import), so no dependency removal is needed there after Task 5 — only the source import is dropped.
