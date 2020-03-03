import argparse

import numpy as np
import torch

from data import SpectrogramDataset, AudioDataLoader
from decoders.decoder import GreedyDecoder
from parsers import add_decoder_args, add_inference_args
from utils import load_model
from ascii_graph import Pyasciigraph

parser = argparse.ArgumentParser(description='DeepSpeech transcription')
parser = add_inference_args(parser)
parser.add_argument('--test-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/test_manifest.csv')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for testing')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")
parser.add_argument('--save-output', default=None, help="Saves output of model from test to this file_path")
parser = add_decoder_args(parser)


def evaluate(test_loader, device, model, decoder, target_decoder, save_output=False, verbose=False, half=False,
             output_file='evaluate.txt', main_proc=True):
    model.eval()
    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
    output_data = []
    min_str, max_str, last_str, min_cer, max_cer = "", "", "", 100, 0
    hcers = dict([(k, 1) for k in range(11)])
    for i, (data) in enumerate(test_loader):
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

        out, output_sizes = model(inputs, input_sizes)

        decoded_output, _ = decoder.decode(out, output_sizes)
        target_strings = target_decoder.convert_to_strings(split_targets)

        if save_output is not None:
            # add output to data array, and continue
            output_data.append((out.cpu().numpy(), output_sizes.numpy(), target_strings))
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
            hcers[int(cer_inst//10)]+=1
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


    if main_proc:
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


if __name__ == '__main__':
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if args.cuda else "cpu")
    model = load_model(device, args.model_path, args.half)

    if args.decoder == "beam":
        from decoders.decoder import BeamCTCDecoder

        decoder = BeamCTCDecoder(model.labels, lm_path=args.lm_path, alpha=args.alpha, beta=args.beta,
                                 cutoff_top_n=args.cutoff_top_n, cutoff_prob=args.cutoff_prob,
                                 beam_width=args.beam_width, num_processes=args.lm_workers)
    elif args.decoder == "greedy":
        decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
    else:
        decoder = None
    target_decoder = GreedyDecoder(model.labels, blank_index=model.labels.index('_'))
    test_dataset = SpectrogramDataset(audio_conf=model.audio_conf, manifest_filepath=args.test_manifest,
                                      labels=model.labels, normalize=True)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)
    wer, cer, output_data = evaluate(test_loader=test_loader,
                                     device=device,
                                     model=model,
                                     decoder=decoder,
                                     target_decoder=target_decoder,
                                     save_output=args.save_output,
                                     verbose=args.verbose,
                                     half=args.half)

    # print('Test Summary \t'
    #       'Average WER {wer:.3f}\t'
    #       'Average CER {cer:.3f}\t'.format(wer=wer, cer=cer))
    if args.save_output is not None:
        np.save(args.save_output, output_data)
