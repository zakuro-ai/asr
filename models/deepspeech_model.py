import os
from apex.parallel import DistributedDataParallel
from apex import amp
from modules.deepspeech import DeepSpeech
from loggers import VisdomLogger, TensorBoardLogger
from decoders import GreedyDecoder
import torch
import json
from utils import supported_rnns
import torch.distributed as dist

class DeepSpeechModel:
    def __init__(self, args):
        if args.gpu_rank:
            torch.cuda.set_device(int(args.gpu_rank))
        # Instantiate the default variables
        [self._loss_results, self._cer_results, self._wer_results] = [torch.Tensor(args.epochs)] * 3
        [self._avg_loss, self._start_epoch, self._start_iter] = [0] * 3
        [self._optim_state, self._amp_state, self._best_cer, self._best_wer, self._wer, self._cer] = [None] * 6

        # Set variables from args
        self.device = torch.device("cuda" if args.cuda else "cpu")
        self._distributed = args.world_size > 1
        self._rank = args.rank
        self._save_folder = args.save_folder
        self._visdom, self._tensorboard = args.visdom, args.tensorboard

        # Instantiate the network
        self.net = self.continue_from(args) if os.path.exists(args.continue_from) else self.instantiate_network(args)

        # Distribute
        self._main_proc = self.distribute(args) if self._distributed else True

        # Boards
        self._visdom_logger = VisdomLogger(args.id, args.epochs) if self._main_proc and self._visdom else None
        self._tensorboard_logger = TensorBoardLogger(args.id,
                                               args.log_dir,
                                               args.log_params) if self._main_proc and  self._tensorboard else None

        os.makedirs(self._save_folder, exist_ok=True)

    def distribute(self, args):
        labels = self.net.labels
        audio_conf = self.net.audio_conf
        main_proc = self.initialize_process_group(args)
        if main_proc and self._visdom:  # Add previous scores to visdom graph
            self._visdom_logger.load_previous_values(self._start_epoch, self._package)
        if main_proc and self._tensorboard:  # Previous scores to tensorboard logs
            self._tensorboard_logger.load_previous_values(self._start_epoch, self._package)
        self.net = DistributedDataParallel(self.net)

        self.net.decoder = GreedyDecoder(labels)
        self.net.audio_conf = audio_conf
        self.net.labels = labels
        self.net.device = self.device
        self.net = self.net.to(self.device)
        return main_proc

    def instantiate_network(self, args):
        with open(args.labels_path) as label_file:
            labels = str(''.join(json.load(label_file)))

        audio_conf = dict(sample_rate=args.sample_rate,
                          window_size=args.window_size,
                          window_stride=args.window_stride,
                          window=args.window,
                          noise_dir=args.noise_dir,
                          noise_prob=args.noise_prob,
                          noise_levels=(args.noise_min, args.noise_max))

        rnn_type = args.rnn_type.lower()
        assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"
        self.net =  DeepSpeech(rnn_hidden_size=args.hidden_size,
                          nb_layers=args.hidden_layers,
                          labels=labels,
                          rnn_type=supported_rnns[rnn_type],
                          audio_conf=audio_conf,
                          bidirectional=args.bidirectional)
        self.init_optimizer(args)
        self.net.decoder = GreedyDecoder(labels)
        self.net.audio_conf = audio_conf
        self.net.labels = labels
        self.net.device = self.device
        self.net = self.net.to(self.device)
        return self.net


    def continue_from(self, args):
        print("Loading checkpoint model %s" % args.continue_from)
        package = torch.load(args.continue_from, map_location=lambda storage, loc: storage)
        self.net = DeepSpeech.load_model_package(package)
        self.init_optimizer(args)
        self._optim_state = package['optim_dict']
        self._amp_state = package['amp']
        self._optimizer.load_state_dict(self._optim_state)
        amp.load_state_dict(self._amp_state)
        self._start_epoch = int(package.get('epoch', 1)) - 1  # Index start at 0 for training
        self._start_iter = package.get('iteration', None)
        if self._start_iter is None:
            self._start_epoch += 1  # We saved model after epoch finished, start at the next epoch.
            self._start_iter = 0
        else:
            self._start_iter += 1
        self._avg_loss = int(package.get('avg_loss', 0))
        loss_results, cer_results, wer_results = package['loss_results'], package['cer_results'], \
                                                 package['wer_results']
        for k, (loss, cer, wer) in enumerate(zip(loss_results, cer_results, wer_results)):
            try:
                self._loss_results[k], self._cer_results[k], self._wer_results[k] = loss, cer, wer
            except IndexError:
                break

        if self._start_epoch>0:
            self._best_cer = min(cer_results[:self._start_epoch])
            self._best_wer = min(wer_results[:self._start_epoch])
            self._cer = cer_results[self._start_epoch - 1]
            self._wer = wer_results[self._start_epoch - 1]
        else:
            self._best_cer = None
            self._best_wer = None
            self._cer = None
            self._wer = None
        self._package = package
        self.net.decoder = GreedyDecoder(self.net.labels)
        self.net.device = self.device
        return self.net

    def init_optimizer(self, args):
        self.net = self.net.to(self.device)
        self._optimizer = torch.optim.SGD(self.net.parameters(),
                                           lr=args.lr,
                                           momentum=args.momentum,
                                           nesterov=True,
                                           weight_decay=1e-5)

        self.net, self._optimizer = amp.initialize(self.net,
                                                   self._optimizer,
                                                   opt_level=args.opt_level,
                                                   keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                                   loss_scale=args.loss_scale)



    @staticmethod
    def initialize_process_group(args):
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        main_proc = args.rank == 0  # Only the first proc should save models
        return main_proc