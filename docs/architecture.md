# Architecture

## Model — DeepSpeech2

`asr_deepspeech.modules.DeepSpeech` implements the DeepSpeech2 architecture:

```
Input spectrogram  (B, 1, freq, time)
        │
   MaskConv          — 2× Conv2d + BatchNorm + Hardtanh, masks padded frames
        │
  Reshape + Transpose — (T, B, H)
        │
   N× BatchRNN        — BiLSTM / BiGRU / RNN, with optional BatchNorm between layers
        │
  [Lookahead]         — only when bidirectional=False
        │
   SequenceWise FC    — BatchNorm1d + Linear → num_classes
        │
  InferenceBatchSoftmax
        │
Output  (B, T, num_classes)
```

Training uses CTC loss (`torch.nn.CTCLoss`). Decoding is greedy argmax by default; beam search with optional LM rescoring is available via `GreedyDecoder` / `BeamDecoder`.

## Data pipeline

```
Landing (raw audio/transcripts)
     │  asr-etl
     ▼
Bronze (CSVs: audio_filepath, duration, fq, text, text_size)
     │  SpectrogramDataset
     ▼
Spectrogram cache (disk .pt files, keyed by MD5 of path)
     │  BucketingSampler → AudioDataLoader → _collate_fn
     ▼
Batch (inputs, targets, input_percentages, target_sizes)
```

`BucketingSampler` sorts by duration before bucketing, cutting average padding by ~30-40%.

## Trainer

`DeepSpeechTrainer` inherits from `sakura.ml.SakuraTrainer` and adds:

- **AMP** — `torch.amp.GradScaler("cuda")`, created once, persisted in checkpoints
- **Gradient clipping** — `nn.utils.clip_grad_norm_` with configurable `max_norm`
- **Checkpoint** — saves on best CER; auto-resumes from `model_path`
- **WER/CER** — accumulated as raw edit distances, normalized once at the end of each eval pass

## Multi-GPU

`asr_deepspeech.multiproc` wraps `torchrun` for DDP training. The `DistributedBucketingSampler` handles epoch-based deterministic shuffle across ranks.
