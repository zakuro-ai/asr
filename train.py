from warpctc_pytorch import CTCLoss
from parsers import parser, parse_args
from models import DeepSpeechModel
from trainers import DeepSpeechTrainer

# Arguments related to the training
parser.add_argument('--batch-size', default=200, type=int, help='Batch size for training')
parser.add_argument('--labels-path', help='Contains all characters for transcription')
# Arguments for saving/loading the model
parser.add_argument('--continue-from', help='Continue from checkpoint model')
parser.add_argument('--overwrite', action='store_true', help='Continue from checkpoint model')
parser.add_argument('--save-folder', default='__data__/models', help='Location to save epoch models')
# Simplify the path to manifests for training
parser.add_argument('--manifest', metavar='DIR', help='manifest id use to simplify the command line')
parser.add_argument('--root-manifest', default='__data__/manifests', help='Contains all characters for transcription')


if __name__ == '__main__':
    args = parse_args(parser)
    model = DeepSpeechModel(args)
    trainer = DeepSpeechTrainer(model=model,
                                batch_size=args.batch_size,
                                criterion=CTCLoss(),
                                args=args)
    trainer.run(epochs=args.epochs)
