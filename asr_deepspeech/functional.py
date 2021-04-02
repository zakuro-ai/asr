import os
import subprocess
from tempfile import NamedTemporaryFile
from scipy.io.wavfile import read
import torch.distributed as dist
import numpy as np
import torch
from ascii_graph import Pyasciigraph


def to_np(x):
    return x.cpu().numpy()




def evaluate(test_loader,
             device,
             model,
             decoder,
             target_decoder,
             save_output=False,
             verbose=False,
             half=False,
             output_file='evaluate.txt',
             main_proc=True):
    model.eval()
    total_cer, total_wer, num_tokens, num_chars = 0, 0, 0, 0
    output_data = []
    min_str, max_str, last_str, min_cer, max_cer = "", "", "", 100, 0
    hcers = dict([(k, 1) for k in range(10)])
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


def load_audio(path):
    sample_rate, sound = read(path)
    assert sample_rate==16000
    sound = sound.astype('float32') / 32767  # normalize audio
    #sound = sound.astype('float32') / sample_rate  # normalize audio
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
    return sound



def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    inputs = torch.zeros(minibatch_size, 1, freq_size, max_seqlength)
    input_percentages = torch.FloatTensor(minibatch_size)
    target_sizes = torch.IntTensor(minibatch_size)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0]
        target = sample[1]
        seq_length = tensor.size(1)
        inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
        input_percentages[x] = seq_length / float(max_seqlength)
        target_sizes[x] = len(target)
        targets.extend(target)
    targets = torch.IntTensor(targets)
    return inputs, targets, input_percentages, target_sizes

#
# def _collate_fn(batch, C=501):
#     def func(p):
#         return p[0].size(1)
#
#     batch = sorted(batch, key=lambda sample: sample[0].size(1), reverse=True)
#     longest_sample = max(batch, key=func)[0]
#     N = len(batch)
#     T = longest_sample.size(1)
#
#     inputs = torch.zeros(T, N, C)
#     input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
#     # input_percentages = torch.FloatTensor(N)
#     target_lengths = torch.IntTensor(N)
#     targets = []
#     for x in range(N):
#         sample = batch[x]
#         tensor, target = sample[0], sample[1]
#
#         seq_length = tensor.size(1)
#         inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
#
#         # input_percentages[x] = seq_length / float(T)
#         target_lengths[x] = len(target)
#         targets.extend(target)
#     targets = torch.IntTensor(targets)
#     return inputs, targets, input_lengths, target_lengths


def get_audio_length(path):
    output = subprocess.check_output(['soxi -D \"%s\"' % path.strip()], shell=True)
    return float(output)


def audio_with_sox(path, sample_rate, start_time, end_time):
    """
    crop and resample the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as tar_file:
        tar_filename = tar_file.name
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} trim {} ={} >/dev/null 2>&1".format(path, sample_rate,
                                                                                               tar_filename, start_time,
                                                                                               end_time)
        os.system(sox_params)
        y = load_audio(tar_filename)
        return y


def augment_audio_with_sox(path, sample_rate, tempo, gain):
    """
    Changes tempo and gain of the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as augmented_file:
        augmented_filename = augmented_file.name
        sox_augment_params = ["tempo", "{:.3f}".format(tempo), "gain", "{:.3f}".format(gain)]
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} {} >/dev/null 2>&1".format(path, sample_rate,
                                                                                      augmented_filename,
                                                                                      " ".join(sox_augment_params))
        os.system(sox_params)
        y = load_audio(augmented_filename)
        return y


def load_randomly_augmented_audio(path, sample_rate=16000, tempo_range=(0.85, 1.15),
                                  gain_range=(-6, 8)):
    """
    Picks tempo and gain uniformly, applies it to the utterance by using sox utility.
    Returns the augmented utterance.
    """
    low_tempo, high_tempo = tempo_range
    tempo_value = np.random.uniform(low=low_tempo, high=high_tempo)
    low_gain, high_gain = gain_range
    gain_value = np.random.uniform(low=low_gain, high=high_gain)
    audio = augment_audio_with_sox(path=path, sample_rate=sample_rate,
                                   tempo=tempo_value, gain=gain_value)
    return audio




def to_np(x):
    return x.cpu().numpy()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def reduce_tensor(tensor, world_size, reduce_op_max=False):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.MAX if reduce_op_max is True else dist.reduce_op.SUM)  # Default to sum
    if not reduce_op_max:
        rt /= world_size
    return rt


def check_loss(loss, loss_value):
    """
    Check that warp-ctc loss is valid and will not break training
    :return: Return if loss is valid, and the error in case it is not
    """
    loss_valid = True
    error = ''
    if loss_value == float("inf") or loss_value == float("-inf"):
        loss_valid = False
        error = "WARNING: received an inf loss"
    elif torch.isnan(loss).sum() > 0:
        loss_valid = False
        error = 'WARNING: received a nan loss, setting loss value to 0'
    elif loss_value < 0:
        loss_valid = False
        error = "WARNING: received a negative loss"
    return loss_valid, error


def load_model(device, model_path, use_half):
    from asr_deepspeech.models import DeepSpeechModel
    model = DeepSpeechModel.load_model(model_path)
    model.eval()
    model = model.to(device)
    if use_half:
        model = model.half()
    return model
