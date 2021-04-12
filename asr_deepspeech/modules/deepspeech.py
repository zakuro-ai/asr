import math
from collections import OrderedDict
import json
from asr_deepspeech.decoders import GreedyDecoder
import os
from ascii_graph import Pyasciigraph
from asr_deepspeech.data.loaders import AudioDataLoader
from asr_deepspeech.data.samplers import BucketingSampler
from .blocks import *
from asr_deepspeech.data.dataset import SpectrogramDataset
from argparse import Namespace
from zakuro import hub

class DeepSpeech(nn.Module):
    def __init__(self,
                 audio_conf,
                 decoder,
                 id="asr",
                 label_path=None,
                 labels=None,
                 rnn_type="nn.LSTM",
                 rnn_hidden_size=768,
                 rnn_hidden_layers=5,
                 bidirectional=True,
                 context=20,
                 version='0.0.1',
                 model_path=None,
                 ):
        super(DeepSpeech, self).__init__()
        labels = json.load(open(label_path, "r")) if labels is None else labels
        self.version = version
        self.id =id
        self.decoder, self.audio_conf = decoder, audio_conf
        self.context = context
        self.rnn_hidden_size = rnn_hidden_size
        self.rnn_hidden_layers = rnn_hidden_layers
        self.rnn_type = eval(rnn_type)
        self.labels = labels
        self.bidirectional = bidirectional
        self.sample_rate = self.audio_conf.sample_rate
        self.window_size = self.audio_conf.window_size
        self.num_classes = len(self.labels)
        self.model_path = model_path
        self.build_network()
        self.decoder = GreedyDecoder(self.labels)
        # try:
        #     assert model_path is not None
        #     assert os.path.exists(model_path)
        #     print(f"{self.id}>> Loading {model_path}")
        #     ckpt = Namespace(**torch.load(model_path))
        #     self.load_state_dict(ckpt.state_dict)
        # except:
        #     pass
    def build_network(self):
        self.conv = MaskConv(nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(20, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True),
            nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), padding=(10, 5)),
            nn.BatchNorm2d(32),
            nn.Hardtanh(0, 20, inplace=True)
        ))
        # Based on above convolutions and spectrogram size using conv formula (W - F + 2P)/ S+1
        rnn_input_size = int(math.floor((self.sample_rate * self.window_size) / 2) + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 20 - 41) / 2 + 1)
        rnn_input_size = int(math.floor(rnn_input_size + 2 * 10 - 21) / 2 + 1)
        rnn_input_size *= 32

        rnns = []
        rnn = BatchRNN(input_size=rnn_input_size,
                       hidden_size=self.rnn_hidden_size,
                       rnn_type=self.rnn_type,
                       bidirectional=self.bidirectional,
                       batch_norm=False)
        rnns.append(('0', rnn))
        for x in range(self.rnn_hidden_layers - 1):
            rnn = BatchRNN(input_size=self.rnn_hidden_size,
                           hidden_size=self.rnn_hidden_size,
                           rnn_type=self.rnn_type,
                           bidirectional=self.bidirectional)
            rnns.append(('%d' % (x + 1), rnn))
        self.rnns = nn.Sequential(OrderedDict(rnns))
        self.lookahead = nn.Sequential(
            # consider adding batch norm?
            Lookahead(self.rnn_hidden_size,
                      context=self.context),
            nn.Hardtanh(0, 20, inplace=True)
        ) if not self.bidirectional else None

        fully_connected = nn.Sequential(
            nn.BatchNorm1d(self.rnn_hidden_size),
            nn.Linear(self.rnn_hidden_size, self.num_classes, bias=False)
        )
        self.fc = nn.Sequential(
            SequenceWise(fully_connected),
        )
        self.inference_softmax = InferenceBatchSoftmax()

    def forward(self, x, lengths):
        lengths = lengths.cpu().int()
        output_lengths = self.get_seq_lens(lengths)
        x, _ = self.conv(x, output_lengths)

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxH

        for rnn in self.rnns:
            x = rnn(x, output_lengths)

        if not self.bidirectional:  # no need for lookahead layer in bidirectional
            x = self.lookahead(x)

        x = self.fc(x)
        x = x.transpose(0, 1)
        # identity in training mode, softmax in eval mode
        x = self.inference_softmax(x)
        return x, output_lengths

    def get_loader(self, manifest, batch_size, num_workers):
        dataset = SpectrogramDataset(audio_conf=self.audio_conf,
                                     manifest_filepath=manifest,
                                     labels=self.labels,
                                     normalize=True,
                                     spec_augment=self.audio_conf.spec_augment)
        sampler = BucketingSampler(dataset,
                                   batch_size=batch_size)
        loader = AudioDataLoader(dataset,
                                 num_workers=num_workers,
                                 batch_sampler=sampler)

        sampler.shuffle()
        return loader, sampler

    def __call__(self,
                 loader = None,
                 manifest=None,
                 batch_size=None,
                 device="cuda",
                 num_workers=32,
                 dist=None,
                 verbose=False,
                 half=False,
                 output_file=None,
                 main_proc=True,
                 restart_from=None):
        with torch.no_grad():
            if loader is None:
                loader, sampler = self.get_loader(manifest=manifest,
                                                  batch_size=batch_size,
                                                  num_workers=num_workers)
            # if restart_from is not None:
            #     hub.restart_from(self, restart_from)

            decoder = self.decoder
            target_decoder = self.decoder
            self.eval()
            self.to(device)
            total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
            output_data = []
            min_str, max_str, last_str, min_cer, max_cer = "", "", "", 100, 0
            hcers = dict([(k, 1) for k in range(10)])
            for i, (data) in enumerate(loader):
                inputs, targets, input_percentages, target_sizes = data
                input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
                inputs = inputs.to(device)
                if half:
                    inputs = inputs.half()
                # unflatten targets
                split_targets = []
                offset = 0
                for size in target_sizes:
                    split_targets.append(targets[offset:offset + size])
                    offset += size

                out, output_sizes = self.forward(inputs, input_sizes)

                decoded_output, _ = decoder.decode(out, output_sizes)
                target_strings = target_decoder.convert_to_strings(split_targets)

                if output_file is not None:
                    # add output to data array, and continue
                    output_data.append((out.detach().cpu().numpy(), output_sizes.numpy(), target_strings))
                for x in range(len(target_strings)):
                    transcript, reference = decoded_output[x][0], target_strings[x][0]
                    wer_inst = decoder.wer(transcript, reference)
                    cer_inst = decoder.cer(transcript, reference)
                    total_wer += wer_inst
                    total_cer += cer_inst
                    num_tokens += len(reference.split())
                    num_chars += len(reference.replace(' ', ''))
                    wer_inst = float(wer_inst) / len(reference.split())
                    cer_inst = float(cer_inst) / len(reference.replace(' ', ''))
                    wer_inst = wer_inst * 100
                    cer_inst = cer_inst * 100
                    wer_inst = min(wer_inst, 100)
                    cer_inst = min(cer_inst, 100)
                    hcers[min(int(cer_inst//10), 9)]+=1
                    last_str = f"Ref:{reference.lower()}" \
                               f"\nHyp:{transcript.lower()}" \
                               f"\nWER:{wer_inst}  " \
                               f"- CER:{cer_inst}"
                    if cer_inst < min_cer:
                        min_cer = cer_inst
                        min_str = last_str
                    if cer_inst > max_cer:
                        max_cer = cer_inst
                        max_str = last_str
                    print(last_str) if verbose else None
            wer = float(total_wer) / num_tokens
            cer = float(total_cer) / num_chars

            cers = [(f'{k*10}-{(k*10) + 10}', v-1) for k, v in hcers.items()]

            graph = Pyasciigraph()
            asciihistogram = "\n|".join(graph.graph('CER histogram', cers))


            if main_proc and output_file is not None:
                with open(output_file, "w") as f:
                    f.write("\n".join([
                        f"================= {wer*100:.2f}/{cer*100:.2f} =================",
                        "----- BEST -----",
                        min_str,
                        "----- LAST -----",
                        last_str,
                        "----- WORST -----",
                        max_str,
                        asciihistogram,
                        "=============================================\n"

                    ]))

            return wer * 100, cer * 100, output_data

    def get_seq_lens(self, input_length):
        """
        Given a 1D Tensor or Variable containing integer sequence lengths, return a 1D tensor or variable
        containing the size sequences that will be output by the network.
        :param input_length: 1D Tensor
        :return: 1D Tensor scaled by model
        """
        seq_len = input_length
        for m in self.conv.modules():
            if type(m) == nn.modules.conv.Conv2d:
                seq_len = ((seq_len + 2 * m.padding[1] - m.dilation[1] * (m.kernel_size[1] - 1) - 1) / m.stride[1] + 1)
        return seq_len.int()
