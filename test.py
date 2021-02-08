import torch
from torch import nn
# Target are to be padded
T = 50      # Input sequence length
C = 501      # Number of classes (including blank)
N = 2      # Batch size
S = 30      # Target sequence length of longest target in batch (padding length)
S_min = 10  # Minimum target length, for demonstration purposes

# Initialize random batch of input vectors, for *size = (T,N,C)
# input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
input = torch.randn(T, N, C).detach().requires_grad_()

# Initialize random batch of targets (0 = blank, 1:C = classes)
target = torch.randint(low=1, high=C, size=(N, S), dtype=torch.long)

input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
target_lengths = torch.randint(low=S_min, high=S, size=(N,), dtype=torch.long)
ctc_loss = nn.CTCLoss()
print(input.shape, target.shape, input_lengths.shape, target_lengths.shape)
loss = ctc_loss(input, target, input_lengths, target_lengths)
loss.backward()


# # Target are to be un-padded
# T = 50      # Input sequence length
# C = 20      # Number of classes (including blank)
# N = 16      # Batch size
#
# # Initialize random batch of input vectors, for *size = (T,N,C)
# input = torch.randn(T, N, C).log_softmax(2).detach().requires_grad_()
# input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
#
# # Initialize random batch of targets (0 = blank, 1:C = classes)
# target_lengths = torch.randint(low=1, high=T, size=(N,), dtype=torch.long)
# target = torch.randint(low=1, high=C, size=(sum(target_lengths),), dtype=torch.long)
# ctc_loss = nn.CTCLoss()
# loss = ctc_loss(input, target, input_lengths, target_lengths)
# loss.backward()