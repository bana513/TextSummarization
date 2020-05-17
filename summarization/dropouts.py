# Based on https://github.com/fastai/fastai/blob/master/fastai/text/models/awd_lstm.py
import warnings
from typing import Collection, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class WeightDropout(nn.Module):
    "A module that warps another layer in which some weights will be replaced by 0 during training."

    def __init__(self, module: nn.Module, weight_p: float, layer_names: Collection[str] = ['weight_hh_l0']):
        super().__init__()
        self.module, self.weight_p, self.layer_names = module, weight_p, layer_names
        for layer in self.layer_names:
            # Makes a copy of the weights of the selected layers.
            w = getattr(self.module, layer)
            self.register_parameter(f'{layer}_raw', nn.Parameter(w.data))
            self.module._parameters[layer] = F.dropout(w, p=self.weight_p, training=False)

    def _setweights(self):
        "Apply dropout to the raw weights."
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=self.training)

    def forward(self, *args):
        self._setweights()
        with warnings.catch_warnings():
            # To avoid the warning that comes because the weights aren't flattened.
            warnings.simplefilter("ignore")
            return self.module.forward(*args)

    def reset(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=False)
        if hasattr(self.module, 'reset'): self.module.reset()


# --------------------------------------------------------------

class WeightDropoutStepwise(nn.Module):
    def __init__(self, module: nn.Module, weight_p: float, layer_names: Collection[str] = ['weight_hh_l0']):
        super().__init__()
        self.module, self.weight_p, self.layer_names = module, weight_p, layer_names
        for layer in self.layer_names:
            w = getattr(self.module, layer)
            self.register_parameter(f'{layer}_raw', nn.Parameter(w.data))
            self.module._parameters[layer] = F.dropout(w, p=self.weight_p, training=False)

    def forward(self, *args):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.module.forward(*args)

    def drop_weights(self):
        for layer in self.layer_names:
            raw_w = getattr(self, f'{layer}_raw')
            self.module._parameters[layer] = F.dropout(raw_w, p=self.weight_p, training=self.training)


# --------------------------------------------------------------
def dropout_mask_(batch_size, hidden_dim, p, device):
    return torch.empty((batch_size, 1, hidden_dim), requires_grad=False, device=device).bernoulli_(1 - p).div_(1 - p)


def dropout_mask(x: Tensor, sz: Collection[int], p: float):
    "Return a dropout mask of the same type as `x`, size `sz`, with probability `p` to cancel an element."
    return x.new(*sz).bernoulli_(1 - p).div_(1 - p)


class RNNDropout(nn.Module):
    "Dropout with probability `p` that is consistent on the seq_len dimension."

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.: return x
        m = dropout_mask(x.data, (x.size(0), 1, x.size(2)), self.p)
        return x * m


class EmbeddingDropout(nn.Module):
    "Apply dropout with probabily `embed_p` to an embedding layer `emb`."

    def __init__(self, emb: nn.Module, embed_p: float):
        super().__init__()
        self.emb, self.embed_p = emb, embed_p
        self.pad_idx = self.emb.padding_idx
        if self.pad_idx is None: self.pad_idx = -1

    def forward(self, words: torch.LongTensor, scale: Optional[float] = None) -> Tensor:
        if self.training and self.embed_p != 0:
            size = (self.emb.weight.size(0), 1)
            mask = dropout_mask(self.emb.weight.data, size, self.embed_p)
            masked_embed = self.emb.weight * mask
        else:
            masked_embed = self.emb.weight
        if scale: masked_embed.mul_(scale)
        return F.embedding(words, masked_embed, self.pad_idx, self.emb.max_norm,
                           self.emb.norm_type, self.emb.scale_grad_by_freq, self.emb.sparse)
