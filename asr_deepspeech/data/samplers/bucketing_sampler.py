import numpy as np
from torch.utils.data import Sampler


class BucketingSampler(Sampler):
    """Batches similarly-sized samples together by pre-bucketing the dataset."""

    def __init__(self, data_source, batch_size=1):
        super().__init__()
        self.data_source = data_source
        ids = list(range(len(data_source)))
        self.bins = [ids[i: i + batch_size] for i in range(0, len(ids), batch_size)]

    def __iter__(self):
        for ids in self.bins:
            np.random.shuffle(ids)
            yield ids

    def __len__(self):
        return len(self.bins)

    def shuffle(self):
        np.random.shuffle(self.bins)
