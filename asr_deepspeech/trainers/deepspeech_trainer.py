import time
from tqdm import tqdm
import torch.utils.data.distributed
from apex import amp
from asr_deepspeech.data.dataset import SpectrogramDataset
from asr_deepspeech.data.loaders import AudioDataLoader
from asr_deepspeech.data.samplers import BucketingSampler, DistributedBucketingSampler
from asr_deepspeech import reduce_tensor, check_loss
from datetime import timedelta
import numpy as np
import math
from argparse import Namespace
import os

class DeepSpeechTrainer:
    def __init__(self,
                 model,
                 criterion,
                 log_dir,
                 batch_size,
                 cuda,
                 num_workers,
                 epochs,
                 start_epoch,
                 silent,
                 checkpoint_per_batch,
                 visdom,
                 tensorboard,
                 log_params,
                 finetune,
                 shuffle,
                 seed,
                 train_manifest,
                 val_manifest,
                 output_file,
                 metrics,
                 optim,
                 dist,
                 save_folder=None,
                 continue_from=None):
        device = "cuda" if cuda else "cpu"
        # # Set the criterion
        self.start_iter = 0
        self.epoch_time = 0
        self.lu = 0
        self.best_cer = None
        self.data = None
        self.avg_loss=0
        self.save_folder = save_folder
        self.continue_from = continue_from


        self.dist = dist
        self.optim = optim
        self.epochs = epochs
        self.log_dir = log_dir
        self.silent = silent
        self.checkpoint_per_batch = checkpoint_per_batch
        self.visdom = visdom
        self.tensorboard = tensorboard
        self.log_params = log_params
        self.finetune = finetune
        self.seed = seed


        self.device = device
        self.model = model
        self.model.to(device)
        self.audio_conf = self.model.audio_conf
        self.criterion = criterion
        self.train_manifest = train_manifest
        self.val_manifest = val_manifest
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.start_epoch = start_epoch
        self.epoch = start_epoch
        self.shuffle = shuffle
        self.main_proc = True
        self.output_file = output_file
        self.metrics = metrics
        self.train_loader, self.test_loader, self.train_sampler = self.get_loaders()
        self.optimizer = self.get_optimizer()
        self.load()
        self.show()

    def run(self):
        for self.epoch in range(self.start_epoch, self.epochs):
            assert self.train()
            assert self.eval()
            assert self.update()

    def show(self):
        # print(self.net)
        print("================ VARS ===================")
        print('id:', self.model.id)
        print('distributed:', self.dist)
        print('train_manifest:', self.train_manifest)
        print('val_manifest:', self.val_manifest)
        print('continue_from:', self.continue_from)
        print('save_folder:', self.save_folder)
        print('output_file:', self.output_file)
        print('main_proc:', self.main_proc)
        print("==========================================")

    def get_optimizer(self):
        self.model = self.model.to(self.device)
        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=self.optim.lr,
                                    momentum=self.optim.momentum,
                                    nesterov=self.optim.nesterov,
                                    weight_decay=self.optim.weight_decay)

        self.model, optimizer = amp.initialize(self.model,
                                               optimizer,
                                               opt_level=self.optim.opt_level,
                                               keep_batchnorm_fp32=self.optim.keep_batchnorm_fp32,
                                               loss_scale=self.optim.loss_scale)
        return optimizer

    def get_loaders(self):
        train_dataset, test_dataset = SpectrogramDataset(audio_conf=self.model.audio_conf,
                                                         manifest_filepath=self.train_manifest,
                                                         labels=self.model.labels,
                                                         normalize=True,
                                                         spec_augment=self.model.audio_conf.spec_augment),\
                                      SpectrogramDataset(audio_conf=self.model.audio_conf,
                                                         manifest_filepath=self.val_manifest,
                                                         labels=self.model.labels,
                                                         normalize=True,
                                                         spec_augment=False)
        if not self.dist is not None:
            train_sampler = BucketingSampler(train_dataset,
                                             batch_size=self.batch_size)
        else:
            train_sampler = DistributedBucketingSampler(train_dataset,
                                                        batch_size=self.batch_size,
                                                        num_replicas=self.dist.world_size,
                                                        rank=self.dist.rank)
        train_loader = AudioDataLoader(train_dataset,
                                       num_workers=self.num_workers,
                                       batch_sampler=train_sampler)
        test_loader = AudioDataLoader(test_dataset,
                                      batch_size=self.batch_size,
                                      num_workers=self.num_workers)

        if (self.shuffle and self.start_epoch != 0) or not self.optim.sorta_grad:
            print("Shuffling batches for the following epochs")
            train_sampler.shuffle(self.start_epoch)
        return train_loader, test_loader, train_sampler

    def description(self):
        id = self.model.id
        lr = self.optimizer.param_groups[0]["lr"]
        tdelta = timedelta(seconds=int(self.epoch_time))
        epoch = self.epoch + 1
        epochs = self.epochs
        lu = self.lu + 1
        avg_loss = self.average_loss()
        c = len(self.metrics.wer)>0
        wer = self.metrics.wer[-1] if c else None
        cer = self.metrics.cer[-1] if c else None
        best_wer = min(self.metrics.wer) if c else None
        best_cer = min(self.metrics.cer) if c else None

        __description__ = ' | '.join([
            f'{id} - {tdelta} >> {epoch}/{epochs} ({lu})',
            f'Lr {lr*pow(10,5):.3f}*e-5',
            f'Loss {avg_loss:.4f}' if avg_loss is not None else '',
            f'WER/CER {wer:.2f}/{cer:.2f} - ({best_wer:.2f}/[{best_cer:.2f}])'
            if self.best_cer is not None else ''
        ])
        return __description__

    def average_loss(self):
        try:
            avg_loss = np.mean(np.array(self.metrics.loss)[:self.epoch])
            assert not math.isnan(avg_loss)
            return avg_loss
        except AssertionError:
            pass

    def train(self):
        self.avg_loss=0
        self.manifest = self.train_manifest
        self.model.train()
        self.end, self.epoch_valid = time.time(), False
        start_epoch_time  = time.time()
        for self.iter, (self.data) in tqdm(enumerate(self.train_loader, start=self.start_iter),
                                           total=len(self.train_loader),
                                           desc=self.description()):
            if self.iter == len(self.train_sampler):
                break
            else:
                self.fit()

        self.avg_loss /= len(self.train_sampler)
        self.metrics.loss.append(self.avg_loss)
        self.epoch_time = time.time() - start_epoch_time
        return True #self.epoch_valid

    def fit(self):

        inputs, targets, input_percentages, target_sizes = self.data
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        # measure data loading time
        inputs = inputs.to('cuda')


        out, output_sizes = self.model.forward(inputs, input_sizes)

        out = out.transpose(0, 1)  # TxNxH
        float_out = out.float()  # ensure float32 for loss
        float_out = float_out.log_softmax(2)
        loss = self.criterion(float_out, targets, output_sizes, target_sizes).to('cuda')
        loss = loss / inputs.size(0)  # average the loss by minibatch

        # Check to ensure valid loss was calculated
        self.epoch_valid = max(self.epoch_valid, self.step(loss))
        # Checkpoint if necessary
        # self.checkpoint_batch()
        del loss, out, float_out

    def step(self, loss):
        if self.dist is not None:#distributed:
            loss = loss.to(self.device)
            loss_value = reduce_tensor(loss, self.dist.world_size).item()
        else:
            loss_value = loss.item()

        valid_loss, error = check_loss(loss, loss_value)
        if valid_loss:
            self.optimizer.zero_grad()
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.optim.max_norm)
            self.optimizer.step()
        else:
            print(error)
            print('Skipping grad update')
            return False

        self.avg_loss += loss_value
        return True

    def eval(self):
        with torch.no_grad():
            self.start_iter = 0  # Reset start iteration for next epoch
            wer, cer, _ = self.model(loader=self.test_loader,
                                     output_file=self.output_file)
            self.metrics.wer.append(wer)
            self.metrics.cer.append(cer)
            return True

    def update(self, save=True):
        # Update the board
        values = {
            'loss_results': self.metrics.loss,
            'cer_results': self.metrics.cer,
            'wer_results': self.metrics.wer
        }
        if self.visdom and self.main_proc:
            self.model.visdom_logger.update(self.epoch, values)
        if self.tensorboard and self.main_proc:
            self.model.tensorboard_logger.update(self.epoch, values, self.model.named_parameters())
            self.model.values = {
                'Avg Train Loss': self.avg_loss,
                'Avg WER': self.metrics.wer[-1],
                'Avg CER': self.metrics.cer[-1],
            }
        condition_best = False
        # Update the conditions the save
        try:
            assert self.best_cer is not None
            assert self.best_cer < self.metrics.cer[-1]
        except AssertionError:
            condition_best = True

        # Save the model
        try:
            assert save
            ckpt_path = f'{self.save_folder}/{self.model.id}-{self.epoch+1}.ckpt.pth'
            files = [(ckpt_path, condition_best), (self.model.model_path, condition_best)]
            for file, c in files:
                self.save(file_path=file) if c else None
        except AssertionError:
            pass

        if condition_best:
            self.best_wer = np.min(self.metrics.wer)
            self.best_cer = np.min(self.metrics.cer)
            self.lu = np.argmin(self.metrics.cer)

        # anneal lr
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / self.optim.learning_anneal

        self.train_sampler.shuffle(self.epoch) if self.shuffle else None

        return True

    def load(self):
        try:
            assert self.continue_from is not None
            assert os.path.exists(self.continue_from)
            ckpt = Namespace(**torch.load(self.continue_from))
            self.metrics = ckpt.metrics
            self.optim = ckpt.optim
            self.optimizer = ckpt.optimizer
            self.model.load_state_dict(ckpt.state_dict)
            self.update(save=False)
            self.start_epoch = self.lu+1
        except:
            pass

    def save(self, file_path):
        torch.save({
            "metrics": self.metrics,
            "optim": self.optim,
            "optimizer": self.optimizer,
            "state_dict": self.model.state_dict()
        }, file_path)
