import argparse
import torch
import numpy as np
from asr_deepspeech.data.dataset import SpectrogramDataset
from asr_deepspeech.data.loaders import AudioDataLoader
from asr_deepspeech.decoders import GreedyDecoder
from asr_deepspeech.parsers import add_decoder_args, add_inference_args
from asr_deepspeech import load_model, evaluate
from asr_deepspeech.decoders import BeamCTCDecoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepSpeech transcription')
    parser = add_inference_args(parser)
    parser.add_argument('--test-manifest', metavar='DIR',
                        help='path to validation manifest csv', default='data/test_manifest.csv')
    parser.add_argument('--batch-size', default=20, type=int, help='Batch size for testing')
    parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--verbose', action="store_true", help="print out decoded output and error of each sample")
    parser.add_argument('--save-output', default=None, help="Saves output of model from test to this file_path")
    parser = add_decoder_args(parser)

    args = parser.parse_args()
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if args.cuda else "cpu")
    model = load_model(device, args.model_path, args.half)

    if args.decoder == "beam":

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
