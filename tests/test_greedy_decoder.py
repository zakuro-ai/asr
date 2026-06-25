import torch
import pytest
from asr_deepspeech.decoders import GreedyDecoder

# Labels: blank=0, then A-E, then space
LABELS = "_ABCDE "


@pytest.fixture
def decoder():
    return GreedyDecoder(LABELS, blank_index=0)


def test_process_string_basic_chars(decoder):
    # indices 1,2,3 → 'A','B','C'
    seq = torch.tensor([1, 2, 3])
    text, offsets = decoder.process_string(seq, 3, remove_repetitions=False)
    assert text == "ABC"
    assert offsets.tolist() == [0, 1, 2]


def test_process_string_strips_blanks(decoder):
    # blank at position 0 should be dropped; 1,2 → 'A','B'
    seq = torch.tensor([0, 1, 2])
    text, offsets = decoder.process_string(seq, 3, remove_repetitions=False)
    assert text == "AB"
    assert offsets.tolist() == [1, 2]


def test_process_string_removes_repetitions(decoder):
    # [1, 1, 2] with remove_repetitions=True → 'A','B' (second A is skipped)
    seq = torch.tensor([1, 1, 2])
    text, offsets = decoder.process_string(seq, 3, remove_repetitions=True)
    assert text == "AB"


def test_process_string_keeps_repetitions_when_disabled(decoder):
    # [1, 1, 2] with remove_repetitions=False → 'A','A','B'
    seq = torch.tensor([1, 1, 2])
    text, offsets = decoder.process_string(seq, 3, remove_repetitions=False)
    assert text == "AAB"


def test_process_string_space_char(decoder):
    # index 6 is space in LABELS
    seq = torch.tensor([1, 6, 2])
    text, offsets = decoder.process_string(seq, 3, remove_repetitions=False)
    assert text == "A B"


def test_decode_argmax_path(decoder):
    # probs shape: (1, 3, 7); argmax → [1, 2, 3] → "ABC"
    probs = torch.zeros(1, 3, len(LABELS))
    probs[0, 0, 1] = 1.0  # A
    probs[0, 1, 2] = 1.0  # B
    probs[0, 2, 3] = 1.0  # C
    strings, offsets = decoder.decode(probs)
    assert strings[0][0] == "ABC"


def test_decode_removes_consecutive_repeats(decoder):
    # argmax → [1, 1, 2]; decode collapses to "AB"
    probs = torch.zeros(1, 3, len(LABELS))
    probs[0, 0, 1] = 1.0  # A
    probs[0, 1, 1] = 1.0  # A (repeat)
    probs[0, 2, 2] = 1.0  # B
    strings, _ = decoder.decode(probs)
    assert strings[0][0] == "AB"


def test_decode_removes_blank_tokens(decoder):
    # argmax → [0, 1, 2]; blank at position 0 is stripped → "AB"
    probs = torch.zeros(1, 3, len(LABELS))
    probs[0, 0, 0] = 1.0  # blank
    probs[0, 1, 1] = 1.0  # A
    probs[0, 2, 2] = 1.0  # B
    strings, _ = decoder.decode(probs)
    assert strings[0][0] == "AB"


def test_decode_batch_of_two(decoder):
    probs = torch.zeros(2, 2, len(LABELS))
    probs[0, 0, 1] = 1.0  # A
    probs[0, 1, 2] = 1.0  # B
    probs[1, 0, 3] = 1.0  # C
    probs[1, 1, 4] = 1.0  # D
    strings, _ = decoder.decode(probs)
    assert strings[0][0] == "AB"
    assert strings[1][0] == "CD"


def test_wer_identical_sentences(decoder):
    assert decoder.wer("hello world", "hello world") == 0


def test_wer_one_substitution(decoder):
    assert decoder.wer("hello world", "hello earth") == 1


def test_cer_identical_strings(decoder):
    assert decoder.cer("hello", "hello") == 0


def test_cer_one_char_difference(decoder):
    assert decoder.cer("hello", "helo") == 1
