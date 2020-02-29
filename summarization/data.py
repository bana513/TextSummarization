from summarization.config import Config
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class SummarizationDataset(Dataset):
    def __init__(self, file, batch_size):
        self.batch_size = batch_size

        self.contents = []
        self.summaries = []

        with open(file, "rb") as f:
            self.contents, self.summaries = pickle.load(f)

        self.contents = self.contents[:2000]
        self.summaries = self.summaries[:2000]
        print("Dataset read done.")

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


