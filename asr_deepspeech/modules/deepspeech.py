import math
import os
from collections import OrderedDict

import pandas as pd
import torch
from torch import nn

from asr_deepspeech.data.loaders import get_loader
from asr_deepspeech.decoders import GreedyDecoder
from asr_deepspeech.modules.blocks import (
    BatchRNN,
    InferenceBatchSoftmax,
    Lookahead,
    MaskConv,
    SequenceWise,
)

_RNN_TYPES = {
    "nn.LSTM": nn.LSTM, "lstm": nn.LSTM,
    "nn.GRU":  nn.GRU,  "gru":  nn.GRU,
    "nn.RNN":  nn.RNN,  "rnn":  nn.RNN,
}


class DeepSpeech(nn.Module):
    def __init__(
        self,
        audio_conf,
        label_path: str,
        id: str = "asr",
        rnn_type: str = "nn.LSTM",
        rnn_hidden_size: int = 768,
        rnn_hidden_layers: int = 5,
        bidirectional: bool = True,
        context: int = 20,
        version: str = "0.0.1",
        model_path: str | None = None,
        # legacy params kept for config compatibility — unused
        decoder=None,
        restart_from=None,
    ):
        super().__init__()
        self.version = version
        self.id = id or "asr"
        self.audio_conf = audio_conf
        self.context = context
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_hidden_layers = rnn_hidden_layers

        if rnn_type not in _RNN_TYPES:
            raise ValueError(f"Unknown rnn_type {rnn_type!r}. Choose from: {list(_RNN_TYPES)}")
        self.rnn_type = _RNN_TYPES[rnn_type]

        # labels: char → int (loaded once at init)
        self.labels: dict[str, int] = {
            v: k for k, v in pd.read_csv(label_path).to_dict()["label"].items()
        }
        self.bidirectional = bidirectional
        self.sample_rate: int = self.audio_conf.sample_rate
        self.window_size: float = self.audio_conf.window_size
        self.num_classes: int = len(self.labels)
        self.model_path = model_path
        self._build_network()
        self.decoder = GreedyDecoder(self.labels)

    def _build_network(self):
        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
        ))

        rnn_input_size = int(math.floor((self.sample_rate * self.window_size) / 2) + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size *= 32

        rnns = [("0", BatchRNN(
            input_size=rnn_input_size,
            hidden_size=self.rnn_hidden_size,
            rnn_type=self.rnn_type,
            bidirectional=self.bidirectional,
            batch_norm=False,
        ))]
        for i in range(1, self.rnn_hidden_layers):
            rnns.append((str(i), BatchRNN(
                input_size=self.rnn_hidden_size,
                hidden_size=self.rnn_hidden_size,
                rnn_type=self.rnn_type,
                bidirectional=self.bidirectional,
            )))
        self.rnns = nn.Sequential(OrderedDict(rnns))

        self.lookahead = (
            nn.Sequential(
                Lookahead(self.rnn_hidden_size, context=self.context),
                nn.Hardtanh(0, 20, inplace=True),
            )
            if not self.bidirectional else None
        )

        self.fc = nn.Sequential(SequenceWise(nn.Sequential(
            nn.BatchNorm1d(self.rnn_hidden_size),
            nn.Linear(self.rnn_hidden_size, self.num_classes, bias=False),
        )))
        self.inference_softmax = InferenceBatchSoftmax()

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        lengths = lengths.cpu().int()
        output_lengths = self.get_seq_lens(lengths)
        x, _ = self.conv(x, output_lengths)

        b, c, d, t = x.size()
        x = x.view(b, c * d, t).transpose(1, 2).transpose(0, 1).contiguous()  # T x N x H

        for rnn in self.rnns:
            x = rnn(x, output_lengths)

        if not self.bidirectional:
            x = self.lookahead(x)

        x = self.fc(x)
        x = x.transpose(0, 1)
        x = self.inference_softmax(x)
        return x, output_lengths

    def get_seq_lens(self, input_length: torch.Tensor) -> torch.Tensor:
        seq_len = input_length
        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                seq_len = (
                    seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1
                ) / m.stride[1] + 1
        return seq_len.int()

    def get_loader(self, manifest: str, batch_size: int, num_workers: int, cache_dir=None):
        return get_loader(self.audio_conf, self.labels, manifest, batch_size, num_workers, cache_dir=cache_dir)

    def finetune_from(self, model_path: str, nlayers: int = 1):
        state_dict = self.state_dict()
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        for k, v in ckpt.items():
            if k in state_dict and state_dict[k].shape == v.shape:
                state_dict[k] = v
            else:
                print(f"skipped {k}")
        self.load_state_dict(state_dict)
        if nlayers is not None:
            for p in list(self.parameters())[:-nlayers]:
                p.requires_grad = False
        print(f"finetuned from {model_path} (last {nlayers} layers)")

    def evaluate(
        self,
        loader,
        device: str = "cuda",
        half: bool = False,
        verbose: bool = False,
        output_file: str | None = None,
        main_proc: bool = True,
    ) -> tuple[float, float, list]:
        """Run inference and return (wer%, cer%, raw_output)."""
        decoder = self.decoder
        self.eval()
        self.to(device)

        total_wer_dist = total_cer_dist = 0
        num_tokens = num_chars = 0
        output_data = []
        min_str = max_str = last_str = ""
        min_cer = 100.0
        max_cer = 0.0
        hcers = {k: 0 for k in range(10)}

        with torch.no_grad():
            for inputs, targets, input_percentages, target_sizes in loader:
                input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
                inputs = inputs.to(device)
                if half:
                    inputs = inputs.half()

                split_targets, offset = [], 0
                for size in target_sizes:
                    split_targets.append(targets[offset: offset + size])
                    offset += size

                out, output_sizes = self.forward(inputs, input_sizes)
                decoded_output, _ = decoder.decode(out, output_sizes)
                target_strings = decoder.convert_to_strings(split_targets)

                if output_file is not None:
                    output_data.append((out.detach().cpu().numpy(), output_sizes.numpy(), target_strings))

                for x in range(len(target_strings)):
                    transcript = decoded_output[x][0]
                    reference = target_strings[x][0]

                    # Accumulate raw edit distances; normalize once at the end.
                    wer_dist = decoder.wer(transcript, reference)
                    cer_dist = decoder.cer(transcript, reference)
                    total_wer_dist += wer_dist
                    total_cer_dist += cer_dist
                    ref_words = len(reference.split())
                    ref_chars = len(reference.replace(" ", ""))
                    num_tokens += ref_words
                    num_chars += ref_chars

                    wer_pct = min(wer_dist / max(ref_words, 1) * 100, 100.0)
                    cer_pct = min(cer_dist / max(ref_chars, 1) * 100, 100.0)
                    hcers[min(int(cer_pct // 10), 9)] += 1
                    last_str = f"Ref:{reference.lower()}\nHyp:{transcript.lower()}\nWER:{wer_pct:.1f}  CER:{cer_pct:.1f}"
                    if cer_pct < min_cer:
                        min_cer, min_str = cer_pct, last_str
                    if cer_pct > max_cer:
                        max_cer, max_str = cer_pct, last_str
                    if verbose:
                        print(last_str)

        wer = total_wer_dist / max(num_tokens, 1) * 100
        cer = total_cer_dist / max(num_chars, 1) * 100

        if main_proc and output_file is not None:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            cers = [(f"{k*10}-{k*10+10}%", v) for k, v in hcers.items()]
            with open(output_file, "w") as f:
                f.write("\n".join([
                    f"===== WER:{wer:.2f}%  CER:{cer:.2f}% =====",
                    "----- BEST -----", min_str,
                    "----- LAST -----", last_str,
                    "----- WORST -----", max_str,
                    str(cers), "=" * 50,
                ]))
            print(f"saved output to {output_file}")

        return wer, cer, output_data
