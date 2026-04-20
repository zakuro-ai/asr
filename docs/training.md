# Training guide

## 1. Prepare data

Download your dataset to the `landing` directory, then run ETL:

```bash
export ZAK_ASR_CONFIG=asr_deepspeech/config_sandbox.yml

# JSUT (Japanese)
asr-etl --dataset jsut

# LibriSpeech (English)
asr-etl --dataset librispeech
```

This creates:
- `<bronze>/<id_dst>/train.csv` — training manifest
- `<bronze>/<id_dst>/test.csv` — validation manifest
- `<bronze>/<id_dst>/labels.csv` — character vocabulary

## 2. Train

```bash
# Single GPU
asr-train

# Multi-GPU (e.g. 2 GPUs)
torchrun --nproc_per_node=2 -m asr_deepspeech.multiproc
```

Training resumes automatically if `model_path` already exists.

## 3. Monitor

A text report is written to `output_file` after each epoch:

```
===== WER:98.06%  CER:34.49% =====
----- BEST -----
Ref:良ある人ならそんな風にに話しかけないだろう
Hyp:用ある人ならそんな風にに話しかけないだろう
WER:100.0  CER:4.8
...
```

TensorBoard logs are written if `tensorboard: true` is set.

## 4. Warm the spectrogram cache (optional)

Pre-computing spectrograms on first run is slow. Warm the cache once before training:

```python
from asr_deepspeech import cfg
from asr_deepspeech.data.dataset import SpectrogramDataset

ds = SpectrogramDataset(
    audio_conf=cfg.model.audio_conf,
    manifest_filepath=cfg.loaders.train_manifest,
    labels=cfg.model.label_path,
    cache_dir=cfg.loaders.cache_dir,
)
ds.warm_cache()
```

## 5. Hyperparameter tips

| Concern | Recommendation |
|---|---|
| GPU OOM | Reduce `batch_size` or `rnn_hidden_size` |
| Slow convergence | Increase `lr` or reduce `weight_decay` |
| CTC loss NaN | Already guarded by `check_loss`; usually means very short clips — increase `min_duration` |
| High WER plateau | Enable `spec_augment: true` and/or `speed_volume_perturb: true` |
| Multi-GPU | Set `device: cuda` and use `torchrun` |
