# asr-deepspeech

DeepSpeech2 ASR in PyTorch — English & Japanese, backed by Zakuro AI.

Supports JSUT (Japanese) and LibriSpeech (English). Runs on single or multi-GPU with mixed-precision training, disk-based spectrogram cache, and duration-bucketed batching.

---

## Modules

| Module | Description |
|---|---|
| `asr_deepspeech` | Top-level package; loads config via `ZAK_ASR_CONFIG` |
| `asr_deepspeech.modules` | DeepSpeech2 model (BiLSTM/GRU/RNN + MaskConv) |
| `asr_deepspeech.trainers` | `DeepSpeechTrainer` — AMP, grad clipping, checkpointing |
| `asr_deepspeech.decoders` | Greedy and beam-search CTC decoders |
| `asr_deepspeech.data` | Dataset, loaders, samplers, parsers |
| `asr_deepspeech.etl` | `JSUTDataset`, `LibriSpeechDataset` — build bronze manifests |
| `asr_deepspeech.audio` | `load_audio`, `duration`, Sox/augmentation helpers |
| `asr_deepspeech.metrics` | `asr_metrics()` — loss/WER/CER namespace tree |
| `asr_deepspeech.loggers` | TensorBoard logger |
| `asr_deepspeech.parsers` | `build_parser`, `add_decoder_args` |
| `asr_deepspeech.multiproc` | `torchrun`-backed multi-GPU launcher |

---

## Installation

**Requirements:** Python ≥ 3.10, PyTorch ≥ 2.0, ffmpeg

```bash
git clone https://github.com/zakuro-ai/asr
cd asr

# With uv (recommended)
uv pip install -e ".[dev]"

# Or with pip
pip install -e ".[dev]"
```

All dependencies are declared in `pyproject.toml`. `requirements.txt` has been removed.

---

## Configuration

Config is loaded from `ZAK_ASR_CONFIG` (falls back to the bundled `config.yml`):

```bash
export ZAK_ASR_CONFIG=asr_deepspeech/config_sandbox.yml
```

Key config sections: `trainer`, `loaders`, `optim`, `model`, `inference`, `meta`.

See [`asr_deepspeech/config_sandbox.yml`](asr_deepspeech/config_sandbox.yml) for a full annotated example.

---

## ETL — build manifests

```bash
# JSUT (Japanese)
asr-etl --dataset jsut

# LibriSpeech (English)
asr-etl --dataset librispeech

# Override paths
asr-etl --dataset jsut --landing /path/to/landing --bronze /path/to/bronze
```

Or run directly:

```bash
python -m asr_deepspeech.etl --dataset jsut
```

---

## Training

```bash
# Single GPU
asr-train

# Multi-GPU (2 GPUs)
torchrun --nproc_per_node=2 -m asr_deepspeech.multiproc
```

The trainer saves the best checkpoint (by CER) to `model_path` and resumes automatically if the file exists.

---

## Dev container

A GPU-enabled dev container is provided for VS Code / GitHub Codespaces:

```
.devcontainer/
  Dockerfile         # CUDA 12.4 + Python 3.12 + uv
  devcontainer.json  # mounts ~/data, exposes GPU, installs deps
```

---

## Tests

```bash
# CPU-only (fast, no data required)
pytest -m "not gpu and not slow"

# Full suite
pytest
```

CI runs on every push via `.github/workflows/ci.yml` (ruff, mypy, pytest).

---

## Notebooks

| Notebook | Description |
|---|---|
| `Prepare JSUT dataset.ipynb` | Build JSUT bronze manifests |
| `Prepare LibriSpeech dataset.ipynb` | Build LibriSpeech bronze manifests |
| `test_loader.ipynb` | Smoke-test the data pipeline |

---

## Pretrained model

A Japanese model trained on JSUT achieves **CER ≈ 34%** on the test set.

---

## Acknowledgements

Fork of [SeanNaren/deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch).
Thanks to [Egor](https://github.com/EgorLakomkin) and [Ryan](https://github.com/ryanleary) for their contributions.
