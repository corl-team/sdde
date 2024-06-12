from collections import defaultdict

import numpy as np
import torch

from ..utils import tmp_seed


class BalancedLabelsSampler:
    """Sample labels with probabilities equal to labels frequency."""
    def __init__(self, labels, labels_per_batch, num_batches, seed):
        counts = np.bincount(labels)
        self._probabilities = counts / np.sum(counts)
        self._labels_per_batch = labels_per_batch
        self._num_batches = num_batches
        self.seed = seed

    def __iter__(self):
        with tmp_seed(self.seed):
            batches = [np.random.choice(len(self._probabilities), self._labels_per_batch,
                                        p=self._probabilities, replace=False)
                       for _ in range(self._num_batches)]
        for batch in batches:
            yield list(batch)


class ShuffledClassBalancedBatchSampler(torch.utils.data.Sampler):
    """Sampler which extracts balanced number of samples for each class.

    Args:
        data_source: Source dataset. Labels field must be implemented.
        batch_size: Required batch size.
        samples_per_class: Number of samples for each class in the batch.
            Batch size must be a multiple of samples_per_class.
        uniform: If true, sample labels uniformly. If false, sample labels according to frequency.
    """

    def __init__(self, data_source, batch_size, samples_per_class, seed=0):
        if batch_size > len(data_source):
            raise ValueError("Dataset size {} is too small for batch size {}.".format(
                len(data_source), batch_size))
        if batch_size % samples_per_class != 0:
            raise ValueError("Batch size must be a multiple of samples_per_class, but {} != K * {}.".format(
                batch_size, samples_per_class))

        self._source_len = len(data_source)
        self.seed = seed
        self._batch_size = batch_size
        self._labels_per_batch = self._batch_size // samples_per_class
        self._samples_per_class = samples_per_class
        labels = np.asarray([data_source.getitem(i, only_label=True)["label"] for i in range(len(data_source))])
        self._label_sampler = BalancedLabelsSampler(labels, self._labels_per_batch,
                                                    num_batches=len(self), seed=self.seed)

        by_label = defaultdict(list)
        for i, label in enumerate(labels):
            by_label[label].append(i)
        self._by_label = list(by_label.values())
        if self._labels_per_batch > len(self._by_label):
            raise ValueError("Can't sample {} classes from dataset with {} classes.".format(
                self._labels_per_batch, len(self._by_label)))

    @property
    def batch_size(self):
        return self._batch_size

    def __iter__(self):
        batches = []
        with tmp_seed(self.seed):
            for labels in self._label_sampler:
                batch = []
                for label in labels:
                    batch.extend(np.random.choice(self._by_label[label], size=self._samples_per_class, replace=True))
                batches.append(batch)
        for batch in batches:
            yield batch

    def __len__(self):
        return self._source_len // self._batch_size


class DistributedShuffledClassBalancedBatchSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self, dataset, batch_size, samples_per_class, num_replicas=None, rank=None,
                 shuffle=True, seed=0, drop_last=False):
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)
        self.batch_size = batch_size
        self.samples_per_class = samples_per_class

    def __iter__(self):
        balanced_sampler = ShuffledClassBalancedBatchSampler(
            self.dataset, self.batch_size, self.samples_per_class, seed=self.epoch + self.seed
        )
        batches = list(iter(balanced_sampler))

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(balanced_sampler), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(balanced_sampler)))  # type: ignore[arg-type]

        indices = indices[:(len(indices) // self.num_replicas) * self.num_replicas]

        # subsample
        indices = indices[self.rank:len(indices):self.num_replicas]
        for i in indices:
            yield batches[i]

    def __len__(self):
        return super().__len__() // self.batch_size
