import torch
import torch.distributed as dist


def _collate_fn(batch):
    """Pad a batch of (spectrogram, transcript) pairs to the longest sequence."""
    batch = sorted(batch, key=lambda s: s[0].size(1), reverse=True)
    longest = batch[0][0]
    freq_size, max_seq = longest.size(0), longest.size(1)
    n = len(batch)

    inputs = torch.zeros(n, 1, freq_size, max_seq)
    input_percentages = torch.FloatTensor(n)
    target_sizes = torch.IntTensor(n)
    targets = []

    for i, (tensor, target) in enumerate(batch):
        seq_len = tensor.size(1)
        inputs[i][0].narrow(1, 0, seq_len).copy_(tensor)
        input_percentages[i] = seq_len / float(max_seq)
        target_sizes[i] = len(target)
        targets.extend(target)

    return inputs, torch.tensor(targets, dtype=torch.int32), input_percentages, target_sizes


def reduce_tensor(tensor, world_size, reduce_op_max=False):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.MAX if reduce_op_max else dist.ReduceOp.SUM)
    if not reduce_op_max:
        rt /= world_size
    return rt


def to_np(x):
    return x.cpu().numpy()


def check_loss(loss, loss_value: float) -> tuple[bool, str]:
    """Return (is_valid, error_message) for a CTC loss value."""
    if loss_value in (float("inf"), float("-inf")):
        return False, "WARNING: received an inf loss"
    if torch.isnan(loss).any():
        return False, "WARNING: received a nan loss"
    if loss_value < 0:
        return False, "WARNING: received a negative loss"
    return True, ""
