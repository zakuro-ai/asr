import math
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
    "nn.LSTM": nn.LSTM,
    "lstm": nn.LSTM,
    "nn.GRU": nn.GRU,
    "gru": nn.GRU,
    "nn.RNN": nn.RNN,
    "rnn": nn.RNN,
}


class DeepSpeech(nn.Module):
    def __init__(
        self,
        audio_conf,
        decoder,
        label_path,
        id="asr",
        rnn_type="nn.LSTM",
        rnn_hidden_size=768,
        rnn_hidden_layers=5,
        bidirectional=True,
        context=20,
        version="0.0.1",
        model_path=None,
        restart_from=None,
    ):
        super().__init__()
        self.version = version
        self.id = id
        self.decoder = decoder
        self.audio_conf = audio_conf
        self.context = context
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_hidden_layers = rnn_hidden_layers

        if rnn_type not in _RNN_TYPES:
            raise ValueError(f"Unknown rnn_type {rnn_type!r}. Choose from: {list(_RNN_TYPES)}")
        self.rnn_type = _RNN_TYPES[rnn_type]

        self.labels = {
            v: k
            for k, v in pd.read_csv(label_path).to_dict()["label"].items()
        }
        self.bidirectional = bidirectional
        self.sample_rate = self.audio_conf.sample_rate
        self.window_size = self.audio_conf.window_size
        self.num_classes = len(self.labels)
        self.model_path = model_path
        self._build_network()
        self.decoder = GreedyDecoder(self.labels)

    def _build_network(self):
        self.conv = MaskConv(
            nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
                nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
            )
        )
        rnn_input_size = int(math.floor((self.sample_rate * self.window_size) / 2) + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size *= 32

        rnns = [
            (
                "0",
                BatchRNN(
                    input_size=rnn_input_size,
                    hidden_size=self.rnn_hidden_size,
                    rnn_type=self.rnn_type,
                    bidirectional=self.bidirectional,
                    batch_norm=False,
                ),
            )
        ]
        for x in range(self.rnn_hidden_layers - 1):
            rnns.append((
                str(x + 1),
                BatchRNN(
                    input_size=self.rnn_hidden_size,
                    hidden_size=self.rnn_hidden_size,
                    rnn_type=self.rnn_type,
                    bidirectional=self.bidirectional,
                ),
            ))
        self.rnns = nn.Sequential(OrderedDict(rnns))
        self.lookahead = (
            nn.Sequential(
                Lookahead(self.rnn_hidden_size, context=self.context),
                nn.Hardtanh(0, 20, inplace=True),
            )
            if not self.bidirectional
            else None
        )
        self.fc = nn.Sequential(
            SequenceWise(
                nn.Sequential(
                    nn.BatchNorm1d(self.rnn_hidden_size),
                    nn.Linear(self.rnn_hidden_size, self.num_classes, bias=False),
                )
            )
        )
        self.inference_softmax = InferenceBatchSoftmax()

    def finetune_from(self, model_path, nlayers=1):
        state_dict = self.state_dict()
        _state_dict = torch.load(model_path, map_location="cpu")
        for k, v in _state_dict.items():
            if k in state_dict and state_dict[k].shape == v.shape:
                state_dict[k] = v
            else:
                print(f"skipped {k}: shape mismatch {state_dict.get(k, 'missing')} vs {v.shape}")
        self.load_state_dict(state_dict)
        print(f"finetuned from {model_path} (last {nlayers} layers)")
        if nlayers is not None:
            for m in list(self.parameters())[:-nlayers]:
                m.requires_grad = False

    def forward(self, x, lengths):
        lengths = lengths.cpu().int()
        output_lengths = self.get_seq_lens(lengths)
        x, _ = self.conv(x, output_lengths)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # T x N x H

        for rnn in self.rnns:
            x = rnn(x, output_lengths)

        if not self.bidirectional:
            x = self.lookahead(x)

        x = self.fc(x)
        x = x.transpose(0, 1)
        x = self.inference_softmax(x)
        return x, output_lengths

    def get_loader(self, manifest, batch_size, num_workers, caching=False):
        return get_loader(
            self.audio_conf,
            self.labels,
            manifest,
            batch_size,
            num_workers,
            caching=caching,
        )

    def __call__(
        self,
        loader=None,
        manifest=None,
        batch_size=None,
        device="cuda",
        num_workers=32,
        verbose=False,
        half=False,
        output_file=None,
        main_proc=True,
        **_,
    ):
        with torch.no_grad():
            if loader is None:
                loader, _ = self.get_loader(
                    manifest=manifest, batch_size=batch_size, num_workers=num_workers
                )

            decoder = self.decoder
            self.eval()
            self.to(device)
            total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
            output_data = []
            min_str, max_str, last_str = "", "", ""
            min_cer, max_cer = 100.0, 0.0
            hcers = {k: 0 for k in range(10)}

            for data in loader:
                inputs, targets, input_percentages, target_sizes = data
                input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
                inputs = inputs.to(device)
                if half:
                    inputs = inputs.half()

                split_targets = []
                offset = 0
                for size in target_sizes:
                    split_targets.append(targets[offset: offset + size])
                    offset += size

                out, output_sizes = self.forward(inputs, input_sizes)
                decoded_output, _ = decoder.decode(out, output_sizes)
                target_strings = decoder.convert_to_strings(split_targets)

                if output_file is not None:
                    output_data.append(
                        (out.detach().cpu().numpy(), output_sizes.numpy(), target_strings)
                    )

                for x in range(len(target_strings)):
                    transcript = decoded_output[x][0]
                    reference = target_strings[x][0]
                    wer_inst = min(decoder.wer(transcript, reference) / max(len(reference.split()), 1) * 100, 100)
                    cer_inst = min(decoder.cer(transcript, reference) / max(len(reference.replace(" ", "")), 1) * 100, 100)
                    total_wer += wer_inst
                    total_cer += cer_inst
                    num_tokens += len(reference.split())
                    num_chars += len(reference.replace(" ", ""))
                    hcers[min(int(cer_inst // 10), 9)] += 1
                    last_str = (
                        f"Ref:{reference.lower()}\nHyp:{transcript.lower()}"
                        f"\nWER:{wer_inst:.1f}  CER:{cer_inst:.1f}"
                    )
                    if cer_inst < min_cer:
                        min_cer, min_str = cer_inst, last_str
                    if cer_inst > max_cer:
                        max_cer, max_str = cer_inst, last_str
                    if verbose:
                        print(last_str)

            wer = total_wer / max(num_tokens, 1)
            cer = total_cer / max(num_chars, 1)

            if main_proc and output_file is not None:
                import os
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                cers = [(f"{k*10}-{k*10+10}", v) for k, v in hcers.items()]
                with open(output_file, "w") as f:
                    f.write(
                        "\n".join([
                            f"===== WER:{wer:.2f}  CER:{cer:.2f} =====",
                            "----- BEST -----", min_str,
                            "----- LAST -----", last_str,
                            "----- WORST -----", max_str,
                            str(cers),
                            "=" * 50,
                        ])
                    )
                print(f"saved output to {output_file}")

            return wer, cer, output_data

    def get_seq_lens(self, input_length):
        seq_len = input_length
        for m in self.conv.modules():
            if isinstance(m, nn.Conv2d):
                seq_len = (
                    seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1
                ) / m.stride[1] + 1
        return seq_len.int()
