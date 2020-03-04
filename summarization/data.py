import pickle

import torch
from torch.utils.data import Dataset, DataLoader

from summarization.config import Config
from summarization.sampler import NoisySortedBatchSampler


def read_dataset(file):
    with open(file, "rb") as f:
        contents, summaries = pickle.load(f)
    return contents, summaries


def split_dataset(contents, summaries, split=.9):
    assert len(contents) == len(summaries)
    split_id = int(len(contents) * split)
    return contents[:split_id], summaries[:split_id], contents[split_id:], summaries[split_id:]


class SummarizationDataset(Dataset):
    def __init__(self, contents, summaries, batch_size):
        self.batch_size = batch_size
        self.contents = contents
        self.summaries = summaries

    def __getitem__(self, idx):
        return torch.LongTensor(self.contents[idx]), torch.LongTensor(self.summaries[idx])

    def __len__(self):
        return len(self.contents)


def collate(batch):
    contents, summaries = zip(*batch)

    padded_contents, content_sizes = pad_sequence(contents, Config.PAD_ID)
    padded_summaries, summary_sizes = pad_sequence(summaries, Config.PAD_ID)

    return padded_contents, content_sizes, padded_summaries, summary_sizes


# based on pytorch's pad_sequence
def pad_sequence(sequences, padding_value=0):
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]

    lengths = [s.size(0) for s in sequences]
    max_len = max(lengths)

    out_dims = (len(sequences), max_len) + trailing_dims
    out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        out_tensor[i, :length, ...] = tensor
    return out_tensor, torch.LongTensor(lengths)


def get_data_loader(contents, summaries, train_set=True):
    dataset = SummarizationDataset(contents, summaries, Config.batch_size)
    sampler = NoisySortedBatchSampler(dataset,
                                      batch_size=Config.batch_size if train_set else 2 * Config.batch_size,
                                      drop_last=True,
                                      shuffle=True if train_set else False,
                                      sort_key_noise=0.02 if train_set else 0)
    loader = DataLoader(dataset,
                        collate_fn=collate,
                        num_workers=0,
                        batch_sampler=sampler)
    return loader
