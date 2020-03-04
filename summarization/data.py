import pickle

import torch
from torch.nn.utils.rnn import pad_sequence
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

        assert len(contents) == len(summaries)
        self.contents = contents
        self.summaries = summaries

    def __getitem__(self, idx):
        return torch.LongTensor(self.contents[idx]), torch.LongTensor(self.summaries[idx])

    def __len__(self):
        return len(self.contents)


def collate(batch):
    contents, summaries = zip(*batch)

    content_sizes = torch.tensor([c.shape[0] for c in contents])
    padded_contents = pad_sequence(contents, batch_first=True, padding_value=Config.PAD_ID)
    summary_sizes = torch.tensor([s.shape[0] for s in summaries])
    padded_summaries = pad_sequence(summaries, batch_first=True, padding_value=Config.PAD_ID)

    return padded_contents, content_sizes, padded_summaries, summary_sizes


def get_data_loader(contents, summaries, train_set=True):
    dataset = SummarizationDataset(contents, summaries, Config.batch_size)
    sampler = NoisySortedBatchSampler(dataset,
                                      batch_size=Config.batch_size if train_set else 2 * Config.batch_size,
                                      drop_last=True,
                                      shuffle=True if train_set else False,
                                      sort_key_noise=0.02 if train_set else 0)
    loader = DataLoader(dataset,
                        collate_fn=collate,
                        num_workers=0,  # https://github.com/pytorch/pytorch/issues/13246
                        batch_sampler=sampler)
    return loader
