import Levenshtein as Lev


class Decoder:
    """Base decoder. Subclasses must implement decode().

    Args:
        labels: dict mapping char → int (as produced by the label CSV loader).
        blank_index: CTC blank token index (default 0).
    """

    def __init__(self, labels: dict, blank_index: int = 0):
        self.labels = labels
        self.int_to_char = {v: k for k, v in labels.items()}
        self.blank_index = blank_index
        self.space_index = labels.get(" ", len(labels))

    def wer(self, s1: str, s2: str) -> int:
        """Word Error Rate as edit distance on word-encoded strings."""
        vocab = set(s1.split()) | set(s2.split())
        word2char = {w: chr(i) for i, w in enumerate(vocab)}
        w1 = "".join(word2char[w] for w in s1.split())
        w2 = "".join(word2char[w] for w in s2.split())
        return Lev.distance(w1, w2)

    def cer(self, s1: str, s2: str) -> int:
        """Character Error Rate as edit distance ignoring spaces."""
        return Lev.distance(s1.replace(" ", ""), s2.replace(" ", ""))

    def decode(self, probs, sizes=None):
        raise NotImplementedError
