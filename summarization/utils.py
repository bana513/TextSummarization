import torch


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths)
    ids = torch.arange(0, max_len, device=lengths.device).long()
    return ids >= lengths.unsqueeze(1)
