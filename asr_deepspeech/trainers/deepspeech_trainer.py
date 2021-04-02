import time
from tqdm import tqdm
import torch.utils.data.distributed
from apex import amp
from asr_deepspeech.data.dataset import SpectrogramDataset
from asr_deepspeech.data.loaders import AudioDataLoader
from asr_deepspeech.data.samplers import BucketingSampler, DistributedBucketingSampler
from asr_deepspeech import evaluate
from asr_deepspeech import reduce_tensor, check_loss
from asr_deepspeech.trainers.asr_trainer import ASRTrainer
from asr_deepspeech.modules import DeepSpeech
import json

class DeepSpeechTrainer(ASRTrainer):
    def __init__(self, model,batch_size, criterion, args):
        super(DeepSpeechTrainer, self).__init__(model, args)
        # Set the criterion
        self.criterion = criterion
        self.speed_volume_perturb = args.speed_volume_perturb
        self.spec_augment = args.spec_augment
        self.no_shuffle = args.no_shuffle
        self.no_sorta_grad = args.no_sorta_grad
        self.max_norm =args.max_norm
        self.data_loaders(batch_size)
        # Show the network
        self.show()

    def run(self, epochs):
        for self.epoch in range(self.start_epoch, epochs):
            assert self.train()
            assert self.eval()
            assert self.update()

    def train(self):
        self.net.train()
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
        self.loss_results[self.epoch] = self.avg_loss
        self.epoch_time = time.time() - start_epoch_time
        return True #self.epoch_valid

    def fit(self):

        inputs, targets, input_percentages, target_sizes = self.data
        input_sizes = input_percentages.mul_(int(inputs.size(3))).int()
        # measure data loading time
        inputs = inputs.to('cuda')


        out, output_sizes = self.net(inputs, input_sizes)

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
        if self.distributed:
            loss = loss.to(self.device)
            loss_value = reduce_tensor(loss, self.world_size).item()
        else:
            loss_value = loss.item()

        valid_loss, error = check_loss(loss, loss_value)
        if valid_loss:
            self.optimizer.zero_grad()
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
            torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.max_norm)
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
            self.wer, self.cer, _ = evaluate(test_loader=self.test_loader,
                                             model=self.net,
                                             device=self.device,
                                             decoder=self.net.decoder,
                                             target_decoder=self.net.decoder,
                                             output_file=self.output_file,
                                             main_proc=self.main_proc)
            self.wer_results[self.epoch] = self.wer
            self.cer_results[self.epoch] = self.cer
            return True

    def update(self):
        # Update the board
        values = {
            'loss_results': self.loss_results,
            'cer_results': self.cer_results,
            'wer_results': self.wer_results
        }
        if self.args.visdom and self.main_proc:
            self.net.visdom_logger.update(self.epoch, values)
        if self.args.tensorboard and self.main_proc:
            self.net.tensorboard_logger.update(self.epoch, values, self.net.named_parameters())
            self.net.values = {
                'Avg Train Loss': self.avg_loss,
                'Avg WER': self.wer,
                'Avg CER': self.cer
            }

        # Save the model
        condition_chkpt = self.main_proc and self.args.checkpoint
        condition_best = self.main_proc and (self.best_cer is None or self.best_cer > self.cer)
        if condition_chkpt or condition_best:
            file_path = f'{self.save_folder}/deepspeech_{self.epoch+1}.pth.tar' if condition_chkpt else self.model_path
            self.serialize(file_path=file_path, avg_loss=self.avg_loss)
            if condition_best:
                self.best_wer = self.wer
                self.best_cer = self.cer
                self.lu = self.epoch

        # anneal lr
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr'] / self.args.learning_anneal

        if not self.args.no_shuffle:
            # print("Shuffling batches...")
            self.train_sampler.shuffle(self.epoch)

        return True

    def data_loaders(self, batch_size):
        train_dataset = SpectrogramDataset(audio_conf=self.audio_conf,
                                           manifest_filepath=self.train_manifest,
                                           labels=self.labels,
                                           normalize=True,
                                           speed_volume_perturb=self.speed_volume_perturb,
                                           spec_augment=self.spec_augment)
        test_dataset = SpectrogramDataset(audio_conf=self.audio_conf,
                                          manifest_filepath=self.val_manifest,
                                          labels=self.labels,
                                          normalize=True,
                                          speed_volume_perturb=False,
                                          spec_augment=False)
        if not self.distributed:
            print('BucketingSampler')
            self.train_sampler = BucketingSampler(train_dataset, batch_size=batch_size)
        else:
            print('DistributedBucketingSampler')
            self.train_sampler = DistributedBucketingSampler(train_dataset,
                                                             batch_size=batch_size,
                                                             num_replicas=self.args.world_size,
                                                             rank=self.rank)
        self.train_loader = AudioDataLoader(train_dataset,
                                            num_workers=self.args.num_workers,
                                            batch_sampler=self.train_sampler)
        self.test_loader = AudioDataLoader(test_dataset,
                                           batch_size=self.args.batch_size,
                                           num_workers=self.args.num_workers)

        if (not self.no_shuffle and self.start_epoch != 0) or self.no_sorta_grad:
            print("Shuffling batches for the following epochs")
            self.train_sampler.shuffle(self.start_epoch)

    def checkpoint_batch(self):
        # measure elapsed time
        self.batch_time.update(time.time() - self.end)
        self.end = time.time()
        if self.checkpoint_per_batch > 0 and self.iter > 0 and (self.iter + 1) % self.checkpoint_per_batch == 0 and self.main_proc:
            file_path = f'{self.save_folder}/deepspeech_checkpoint_epoch_{self.epoch+1}_iter_{self.iter + 1}.pth'
            print("Saving checkpoint model to %s" % file_path)
            self.serialize(avg_loss=self.avg_loss / self.iter)

    def serialize(self, file_path, avg_loss=None):
        torch.save(DeepSpeech.serialize(self.net,
                                        optimizer=self.optimizer,
                                        amp=amp,
                                        epoch=self.epoch,
                                        iteration=self.iter,
                                        loss_results=self.loss_results,
                                        wer_results=self.wer_results,
                                        cer_results=self.cer_results,
                                        avg_loss=avg_loss),
                   file_path)

