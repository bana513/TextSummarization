import random

from torch.utils.data.sampler import BatchSampler, Sampler


def _identity(e):
    return e


# https://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.samplers.html
class NoisySortedBatchSampler(BatchSampler):
    def __init__(self,
                 dataset,
                 batch_size,
                 drop_last,
                 sort_key=_identity,
                 sort_key_noise=0.25,
                 last_batch_first=True,
                 shuffle=True):
        data = [len(dataset.summaries[i]) for i in range(len(dataset))]
        self.last_batch_first = last_batch_first
        self.shuffle = shuffle
        super().__init__(
            NoisySortedSampler(data=data, sort_key=sort_key, sort_key_noise=sort_key_noise),
            batch_size, drop_last)

    def __iter__(self):
        batches = list(super().__iter__())
        if self.last_batch_first:
            last_batch = batches.pop()
        if self.shuffle:
            random.shuffle(batches)
        if self.last_batch_first:
            batches.insert(0, last_batch)
        return iter(batches)


class NoisySortedSampler(Sampler):
    def __init__(self, data, sort_key=_identity, sort_key_noise=0.25):
        super().__init__(data)
        self.data = data
        self.sort_key = sort_key
        zip_ = []
        for i, row in enumerate(self.data):
            value = self.sort_key(row)
            noise_value = value * sort_key_noise
            noise = random.uniform(-noise_value, noise_value)
            value = noise + value
            zip_.append(tuple([i, value]))
        zip_ = sorted(zip_, key=lambda r: r[1])
        self.sorted_indexes = [item[0] for item in zip_]

    def __iter__(self):
        return iter(self.sorted_indexes)

    def __len__(self):
        return len(self.data)
