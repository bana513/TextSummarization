import random
import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel
from summarization import Config
from summarization import UsedBertTokens
from summarization import get_mask_from_lengths
from summarization.dropouts import WeightDropout, RNNDropout


def shrink_embedding_layer(embedding_layer):
    token_ids = UsedBertTokens.get_instance().token_ids

    # emb_layer = nn.Embedding(len(token_ids), embedding_dim=Config.embedding_dim)
    embedding_layer.weight.data = embedding_layer.weight[token_ids]

    return embedding_layer

class BertSummarizer(nn.Module):
    def __init__(self):
        super().__init__()

        # self.encoder = BertModel.from_pretrained('bert-base-multilingual-cased')
        # for p in self.encoder.parameters():
        #     p.requires_grad = False
        # self.encoder.eval()
        # self.encoder.train()


        bert =  BertModel.from_pretrained('bert-base-multilingual-cased')
        bert.train()
        self.embedding_layer = bert.embeddings.word_embeddings

        self.embedding_layer = shrink_embedding_layer(self.embedding_layer)
        self.embedding_layer.weight.requires_grad = False

        # self.encoder = nn.LSTM(input_size=Config.encoder_dim,
        #                        hidden_size=Config.encoder_dim//2,
        #                        num_layers=3,
        #                        batch_first=True,
        #                        bidirectional=True)
        self.encoder = Encoder()

        # Calculate total reduction to create correct attention plots
        total_reduction = 1
        for cell in self.encoder.encoder:
            total_reduction *= cell.reduction

        self.decoder = Decoder(self.embedding_layer, total_reduction)
        self.decoder.train()

    def forward(self, batch, teacher_forcing_ratio=1.0, max_len=256):
        padded_contents, content_sizes, padded_summaries, summary_sizes = batch

        padded_contents = padded_contents.to(Config.device)
        if padded_summaries is not None:
            padded_summaries = padded_summaries.to(Config.device)

        with torch.no_grad():
            embs = self.embedding_layer(padded_contents)

        features, content_sizes = self.encoder(embs, content_sizes)

        attn_mask = get_mask_from_lengths(content_sizes).to(Config.device)

        output, attentions = self.decoder(features, attn_mask, padded_summaries, teacher_forcing_ratio, max_len)
        return output, attentions


class RnnCell(nn.Module):
    def __init__(self, input_size, hidden_size, reduction=1, proj_size=None, bidirectional=False, h_drop=0, w_drop=0):
        super().__init__()
        self.reduction = reduction
        self.proj_size = proj_size
        self.bidirectional = bidirectional
        self.cell = nn.LSTM(input_size=input_size * reduction,
                            hidden_size=hidden_size,
                            batch_first=True,
                            bidirectional=bidirectional)
        # regularization
        layer_names = ['weight_hh_l0', 'weight_hh_l0_reverse'] if bidirectional else ['weight_hh_l0']
        self.cell = WeightDropout(self.cell, w_drop, layer_names=layer_names)
        self.output_drop = RNNDropout(h_drop)

        if self.proj_size is not None:
            self.proj = nn.Linear(hidden_size * 2 if bidirectional else hidden_size, proj_size)
            nn.init.xavier_uniform_(self.proj.weight, gain=torch.nn.init.calculate_gain("tanh"))

    def forward(self, input_data, input_size, hs=None):
        bs, seq_len, hdim = input_data.shape

        # Reduction
        if self.reduction > 1:
            input_size = input_size/2
            input_data = input_data[:, :self.reduction*(seq_len//self.reduction), :]
            input_data = input_data.reshape(bs, seq_len // self.reduction, hdim * self.reduction)

        packed = pack_padded_sequence(input_data, input_size, batch_first=True, enforce_sorted=False)
        output, hidden = self.cell(packed)
        output, output_size = pad_packed_sequence(output, batch_first=True)

        if self.proj_size is not None:
            output = torch.tanh(self.proj(output))

        return output, output_size, hs


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.ModuleList([RnnCell(input_size=[Config.embedding_dim, 2 * Config.encoder_dim][i],
                                              hidden_size=Config.encoder_dim,
                                              reduction=[2, 2][i],
                                              bidirectional=True,
                                              proj_size=None,  # [Config.encoder_hidden_dim, None][i],
                                              h_drop=Config.hidden_drop,
                                              w_drop=Config.weight_drop) for i in range(2)])
        # Regularization
        self.input_drop = RNNDropout(Config.input_drop)

    def forward(self, features, feature_sizes):
        features = self.input_drop(features)

        for encoder in self.encoder:
            features, feature_sizes, _ = encoder(features, feature_sizes)

        return features, feature_sizes


class Decoder(nn.Module):
    def __init__(self, bert_embedding, total_reduction):
        super().__init__()
        self.decoder = nn.LSTM(input_size=2*Config.encoder_dim + Config.embedding_dim,
                               hidden_size=Config.decoder_dim,
                               batch_first=True)

        self.bert_embedding = bert_embedding
        self.total_reduction = total_reduction

        self.query_transform = nn.Linear(in_features=Config.decoder_dim, out_features=Config.attention_dim)
        self.key_transform = nn.Linear(in_features=2*Config.encoder_dim, out_features=Config.attention_dim)
        self.attention = BahdanauAttention(Config.attention_dim)
        self.decoder_linear = nn.Linear(in_features=Config.decoder_dim, out_features=Config.decoder_token_num)

        # initializations
        nn.init.xavier_uniform_(self.query_transform.weight, gain=torch.nn.init.calculate_gain("tanh"))
        nn.init.xavier_uniform_(self.key_transform.weight, gain=torch.nn.init.calculate_gain("tanh"))
        nn.init.xavier_uniform_(self.decoder_linear.weight, gain=torch.nn.init.calculate_gain("linear"))

    def forward(self, features, attn_mask, padded_summaries, teacher_forcing_ratio, max_len):
        batch_size, seq_len, features_num = features.shape
        prev_word = torch.tensor([Config.S_ID] * batch_size, device=features.device)
        hidden = (torch.zeros((1, batch_size, Config.decoder_dim), device=features.device),
                  torch.zeros((1, batch_size, Config.decoder_dim), device=features.device))

        keys = torch.tanh(self.key_transform(features))

        # Decoding
        prev_words = []
        attentions = []
        if padded_summaries is not None: max_len = padded_summaries.shape[1]
        for s in range(max_len):
            prev_word, hidden, attention = self.step(features, attn_mask, keys, prev_word, hidden)
            prev_words.append(prev_word)
            attentions.append(attention)

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

            if use_teacher_forcing and padded_summaries is not None:
                prev_word = padded_summaries[:, s]
            else:
                prev_word = prev_word.detach().argmax(1)

        attentions = torch.cat(attentions, 1)
        attentions = attentions.repeat_interleave(self.total_reduction, 2)

        output = torch.stack(prev_words, 1)

        return output, attentions

    def step(self, input, mask, key, prev_word, hidden):
        query = torch.tanh(self.query_transform(hidden[0].transpose(0, 1)))
        context, attention = self.attention(q=query, k=key, v=input, mask=mask)

        with torch.no_grad():
            embs = self.bert_embedding(prev_word)

        combined = torch.cat((context, embs), 1).unsqueeze(1)

        output, hidden = self.decoder(combined, hidden)

        # output shape bs x seq_len x emb_dim
        output = self.decoder_linear(output).squeeze(1)
        output = F.log_softmax(output, dim=-1)
        return output, hidden, attention


class BahdanauAttention(nn.Module):
    def __init__(self, attention_dim):
        super().__init__()

        self.energy_linear = nn.Linear(attention_dim, 1, bias=False)
        nn.init.xavier_uniform_(self.energy_linear.weight, gain=torch.nn.init.calculate_gain("linear"))

    def forward(self, q, k, v, mask=None):
        energy = self.energy_linear(torch.tanh(q + k).float()).squeeze(2)
        if mask is not None:
            energy.masked_fill_(mask.squeeze(1), -np.inf)

        attention = F.softmax(energy, dim=1).unsqueeze(1)
        context = torch.bmm(attention.to(v.dtype), v).squeeze(1)
        return context, attention
