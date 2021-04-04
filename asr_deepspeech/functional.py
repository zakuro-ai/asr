
import torch.distributed as dist
import torch
import yaml
from argparse import Namespace


def load_config():
    def unfold(l):
        if type(l)==str:
            return [l]
        L = []
        for v in l:
            if type(v)==list:
                L.extend(unfold(v))
            else:
                L.append(v)
        return L
    ns = yaml.load(open("config.yml", "r"), Loader=yaml.FullLoader)
    for gp, key, sep in [
        ("meta",        "root_data",        "/"),
        ("meta",        "output_file",      "/"),
        ("model",       "id",               ""),
        ("model",       "label_path",       "/"),
        ("model",       "model_path",       "/"),
        ("trainer",     "continue_from",    "/"),
        ("trainer",     "train_manifest",   "/"),
        ("trainer",     "val_manifest",     "/"),
        ("trainer",     "output_file",      "/"),
        ("trainer",     "save_folder",      "/"),
        ("inference",   "manifest",         "/"),
        ("inference",   "output_file",      "/"),

    ]:
        ns[gp][key] = sep.join(unfold(ns[gp][key])) if type(ns[gp][key])==list else ns[gp][key]

    ns = Namespace(**ns)


    ns.dist = Namespace(**ns.dist) if ns.dist is not None else None
    ns.model["decoder"], ns.model["audio_conf"], ns.trainer["metrics"] = \
        Namespace(**ns.model["decoder"]),\
        Namespace(**ns.model["audio_conf"]),\
        Namespace(**ns.trainer["metrics"])
    ns.trainer["optim"] = Namespace(**ns.trainer["optim"])
    return ns


def to_np(x):
    return x.cpu().numpy()




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
