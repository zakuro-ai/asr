import torch

from asr_deepspeech.modules.blocks import MaskConv


def test_maskconv_runs_on_cpu_and_masks_padding():
    seq = torch.nn.Sequential(torch.nn.Conv2d(1, 1, kernel_size=3, padding=1))
    mod = MaskConv(seq)
    x = torch.randn(2, 1, 8, 10)
    lengths = torch.tensor([10, 4])
    out, out_lengths = mod(x, lengths)
    assert out.shape == x.shape
    # Second sample: everything past length 4 in the time dim must be zeroed.
    assert torch.count_nonzero(out[1, 0, :, 4:]) == 0
