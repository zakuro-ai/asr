# Configuration reference

Config files use YAML with `{{ *key }}` anchor interpolation (resolved by `gnutools.fs.load_config`).

Set the active config via the environment variable:
```bash
export ZAK_ASR_CONFIG=asr_deepspeech/config_sandbox.yml
```

Falls back to `asr_deepspeech/config.yml` if unset.

---

## Top-level keys

| Key | Type | Description |
|---|---|---|
| `dst` | str | Dataset name (e.g. `"jsut"`) |
| `lang` | str | Language code (`"jp"`, `"en"`) |
| `root_filestore` | path | Root of the data lake |
| `fq` | int | Target sample rate in Hz (default `16000`) |
| `landing` | path | Raw source data |
| `bronze` | path | Processed manifests and audio symlinks |
| `gold` | path | Trained model checkpoints |
| `var` | path | Output text files |
| `cache` | path | Spectrogram disk cache |
| `min_duration` | float | Minimum clip duration in seconds |
| `max_duration` | float | Maximum clip duration in seconds |
| `label_path` | path | Path to `labels.csv` |
| `model_path` | path | Path to `.pth` checkpoint |

---

## `trainer`

| Key | Type | Description |
|---|---|---|
| `model_path` | path | Where to save/load the best checkpoint |
| `checkpoint_path` | path\|null | Legacy field — unused |
| `output_file` | path | Path for per-epoch WER/CER text report |
| `epochs` | int | Total training epochs |
| `device` | str | Training device (`"cuda"`, `"cpu"`) |
| `device_test` | str | Evaluation device |
| `mixed_precision` | bool | Enable AMP (`torch.amp`) — requires CUDA |
| `overwrite_lr` | float\|null | Override LR loaded from checkpoint |
| `max_norm` | float | Gradient clipping max norm (default `400.0`) |

---

## `loaders`

| Key | Type | Description |
|---|---|---|
| `train_manifest` | path | Training CSV manifest |
| `val_manifest` | path | Validation CSV manifest |
| `batch_size` | int | Samples per batch |
| `num_workers` | int | DataLoader worker count |
| `cache_dir` | path\|null | Spectrogram disk cache directory |

---

## `optim`

AdamW optimizer + StepLR scheduler.

| Key | Type | Description |
|---|---|---|
| `lr` | float | Initial learning rate |
| `eps` | float | AdamW epsilon |
| `betas` | list[float] | AdamW betas `[β1, β2]` |
| `weight_decay` | float | L2 regularisation |
| `step` | int | StepLR step size (epochs) |
| `gamma` | float | StepLR decay factor |

---

## `model`

| Key | Type | Description |
|---|---|---|
| `id` | str | Human-readable model name |
| `label_path` | path | CSV with one `label` column per character |
| `model_path` | path | Checkpoint path (also used for loading) |
| `rnn_hidden_size` | int | Hidden units per RNN layer |
| `rnn_hidden_layers` | int | Number of RNN layers |
| `rnn_type` | str | `"nn.LSTM"`, `"nn.GRU"`, or `"nn.RNN"` |
| `context` | int | Lookahead context frames (unidirectional only) |
| `bidirectional` | bool | Use bidirectional RNN |
| `restart_from` | path\|null | Legacy — use checkpoint resume instead |
| `decoder.beam_width` | int | Beam width for beam decoder |
| `decoder.lm_path` | path\|null | Path to KenLM language model |
| `audio_conf.sample_rate` | int | Expected sample rate of input audio |
| `audio_conf.window_size` | float | STFT window size in seconds |
| `audio_conf.window_stride` | float | STFT hop size in seconds |
| `audio_conf.window` | str | Window function (`"hamming"`, `"hann"`, ...) |
| `audio_conf.speed_volume_perturb` | bool | Enable Sox speed/volume augmentation |
| `audio_conf.spec_augment` | bool | Enable SpecAugment (freq + time masking) |
| `audio_conf.noise_dir` | path\|null | Directory of noise clips for injection |
| `audio_conf.noise_prob` | float | Probability of applying noise |
| `audio_conf.noise_min` | float | Minimum noise SNR |
| `audio_conf.noise_max` | float | Maximum noise SNR |

---

## `inference`

| Key | Description |
|---|---|
| `manifest` | CSV manifest to run inference on |
| `output_file` | Path for the WER/CER + histogram report |
| `cuda` | Run on GPU |
| `half` | FP16 inference |
| `batch_size` | Inference batch size |
| `num_workers` | DataLoader workers |
