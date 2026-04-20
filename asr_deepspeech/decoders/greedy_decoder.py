import torch
from .decoder import Decoder


class GreedyDecoder(Decoder):
    def __init__(self, labels, blank_index=0):
        super().__init__(labels, blank_index)

    def convert_to_strings(self, sequences, sizes=None, remove_repetitions=False, return_offsets=False):
        strings = []
        offsets = [] if return_offsets else None
        for x in range(len(sequences)):
            seq_len = sizes[x] if sizes is not None else len(sequences[x])
            string, string_offsets = self.process_string(sequences[x], seq_len, remove_repetitions)
            strings.append([string])
            if return_offsets:
                offsets.append([string_offsets])
        if return_offsets:
            return strings, offsets
        return strings

    def process_string(self, sequence, size, remove_repetitions=False):
        string = ""
        offsets = []
        for i in range(size):
            char = self.int_to_char[sequence[i].item()]
            if char == self.int_to_char[self.blank_index]:
                continue
            if remove_repetitions and i != 0 and char == self.int_to_char[sequence[i - 1].item()]:
                continue
            string += char
            offsets.append(i)
        return string, torch.tensor(offsets, dtype=torch.int)

    def decode(self, probs, sizes=None):
        """Argmax decoding with blank and repetition removal.

        Args:
            probs: (batch, seq_len, num_classes)
            sizes: sequence lengths per sample
        Returns:
            strings, offsets
        """
        _, max_probs = torch.max(probs, 2)
        strings, offsets = self.convert_to_strings(
            max_probs.view(max_probs.size(0), max_probs.size(1)),
            sizes,
            remove_repetitions=True,
            return_offsets=True,
        )
        return strings, offsets
