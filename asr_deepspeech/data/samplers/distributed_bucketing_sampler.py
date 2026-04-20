import math

import torch
from torch.distributed import get_rank, get_world_size
from torch.utils.data import Sampler


class DistributedBucketingSampler(Sampler):
    """Distributed variant of BucketingSampler — each rank gets a disjoint slice."""

    def __init__(self, data_source, batch_size: int = 1, num_replicas=None, rank=None):
        super().__init__()
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_replicas = num_replicas
        self.rank = rank

        if hasattr(data_source, "df") and "duration" in data_source.df.columns:
            ids = data_source.df["duration"].argsort().tolist()
        else:
            ids = list(range(len(data_source)))
        self.bins = [ids[i: i + batch_size] for i in range(0, len(ids), batch_size)]

        self.num_samples = math.ceil(len(self.bins) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        bins = self.bins + self.bins[: self.total_size - len(self.bins)]
        assert len(bins) == self.total_size
        return iter(bins[self.rank:: self.num_replicas])

    def __len__(self):
        return self.num_samples

    def shuffle(self, epoch: int = 0):
        g = torch.Generator()
        g.manual_seed(epoch)
        order = torch.randperm(len(self.bins), generator=g).tolist()
        self.bins = [self.bins[i] for i in order]
