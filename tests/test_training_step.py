# tests/test_training_step.py
import torch
import torch.nn as nn

from asr_deepspeech.modules.deepspeech import DeepSpeech

FREQ_BINS = 161


def _model(label_csv, audio_conf):
    return DeepSpeech(
        audio_conf=audio_conf,
        decoder=None,
        label_path=str(label_csv),
        rnn_type="nn.GRU",
        rnn_hidden_size=32,
        rnn_hidden_layers=1,
        bidirectional=True,
    )


def test_ctc_train_step_loss_decreases_on_cpu(label_csv, audio_conf):
    torch.manual_seed(0)
    model = _model(label_csv, audio_conf).to("cpu")
    model.train()

    inputs = torch.randn(2, 1, FREQ_BINS, 40)
    targets = torch.randint(1, 26, (2, 3), dtype=torch.int32)
    flat_targets = targets.reshape(-1)
    target_sizes = torch.tensor([3, 3], dtype=torch.int32)
    input_percentages = torch.ones(2)

    criterion = nn.CTCLoss(blank=0, reduction="sum", zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    def step():
        input_sizes = input_percentages.mul(int(inputs.size(3))).int()
        out, output_sizes = model.forward(inputs, input_sizes)
        out = out.transpose(0, 1).float().log_softmax(2)
        return criterion(out, flat_targets, output_sizes, target_sizes) / inputs.size(0)

    first = None
    for i in range(40):
        opt.zero_grad()
        loss = step()
        loss.backward()
        opt.step()
        if i == 0:
            first = loss.item()
    assert loss.item() < first  # the model is learning
