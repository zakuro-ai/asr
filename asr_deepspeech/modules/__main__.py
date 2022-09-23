import torch
from .deepspeech import DeepSpeech


if __name__ == '__main__':
    import os.path
    import argparse

    parser = argparse.ArgumentParser(description='DeepSpeech model information')
    parser.add_argument('--model-path', default='models/deepspeech_final.pth',
                        help='Path to model file created by training')
    args = parser.parse_args()
    package = torch.load(args.model_path, map_location=lambda storage, loc: storage)
    model = DeepSpeech.load_model(args.model_path)

    print("Model name:         ", os.path.basename(args.model_path))
    print("DeepSpeech version: ", model.version)
    print("")
    print("Recurrent Neural Network Properties")
    print("  RNN Type:         ", model.rnn_type.__name__.lower())
    print("  RNN Layers:       ", model.hidden_layers)
    print("  RNN Size:         ", model.hidden_size)
    print("  Classes:          ", len(model.labels))
    print("")
    print("Model Features")
    print("  Labels:           ", model.labels)
    print("  Sample Rate:      ", model.audio_conf.get("sample_rate", "n/a"))
    print("  Window Type:      ", model.audio_conf.get("window", "n/a"))
    print("  Window Size:      ", model.audio_conf.get("window_size", "n/a"))
    print("  Window Stride:    ", model.audio_conf.get("window_stride", "n/a"))

    if package.get('loss_results', None) is not None:
        print("")
        print("Training Information")
        epochs = package['epoch']
        print("  Epochs:           ", epochs)
        print("  Current Loss:      {0:.3f}".format(package['loss_results'][epochs - 1]))
        print("  Current CER:       {0:.3f}".format(package['cer_results'][epochs - 1]))
        print("  Current WER:       {0:.3f}".format(package['wer_results'][epochs - 1]))
