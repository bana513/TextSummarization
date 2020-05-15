import random
import torch
import torch.nn.functional as F
import numpy as np

from torch import nn
from transformers import BertModel
from summarization import Config
from summarization import UsedBertTokens
from summarization import get_mask_from_lengths


def shrink_embedding_layer(embedding_layer):
    token_ids = UsedBertTokens.get_instance().token_ids
    emb_layer = nn.Embedding(len(token_ids), embedding_dim=Config.encoder_dim)
    # embedding_layer.weight.data = embedding_layer.weight[token_ids]
    return emb_layer

class BertSummarizer(nn.Module):
    def __init__(self):
        super().__init__()

        # self.encoder = BertModel.from_pretrained('bert-base-multilingual-cased')
        # for p in self.encoder.parameters():
        #     p.requires_grad = False
        # self.encoder.eval()
        # self.encoder.train()


        # bert =  BertModel.from_pretrained('bert-base-multilingual-cased')
        # bert.train()
        # self.bert_embedding_layer = bert.embeddings.word_embeddings

        self.bert_embedding_layer = shrink_embedding_layer(None)

        self.encoder = nn.LSTM(input_size=Config.encoder_dim,
                               hidden_size=Config.encoder_dim//2,
                               num_layers=1,
                               batch_first=True,
                               bidirectional=True)

        self.decoder = Decoder(self.bert_embedding_layer)
        self.decoder.train()

    def forward(self, batch, teacher_forcing_ratio=1.0, max_len=256):
        padded_contents, content_sizes, padded_summaries, summary_sizes = batch

        padded_contents = padded_contents.to(Config.device)
        if padded_summaries is not None:
            padded_summaries = padded_summaries.to(Config.device)

        attn_mask = get_mask_from_lengths(content_sizes).to(Config.device)

        # with torch.no_grad():
        embs = self.bert_embedding_layer(padded_contents) # TODO
        features, _ = self.encoder(embs) # attention_mask=attn_mask

        output, attentions = self.decoder(features, attn_mask, padded_summaries, teacher_forcing_ratio, max_len)
        return output, attentions


class Decoder(nn.Module):
    def __init__(self, bert_embedding):
        super().__init__()
        self.decoder = nn.LSTM(input_size=Config.encoder_dim*2,
                               hidden_size=Config.decoder_hidden_dim,
                               batch_first=True)

        self.bert_embedding = bert_embedding

        self.query_transform = nn.Linear(in_features=Config.decoder_hidden_dim, out_features=Config.attention_dim)
        self.key_transform = nn.Linear(in_features=Config.encoder_dim, out_features=Config.attention_dim)
        self.attention = BahdanauAttention(Config.attention_dim)
        self.decoder_linear = nn.Linear(in_features=Config.decoder_hidden_dim, out_features=Config.decoder_token_num)

        # initializations
        nn.init.xavier_uniform_(self.query_transform.weight, gain=torch.nn.init.calculate_gain("tanh"))
        nn.init.xavier_uniform_(self.key_transform.weight, gain=torch.nn.init.calculate_gain("tanh"))
        nn.init.xavier_uniform_(self.decoder_linear.weight, gain=torch.nn.init.calculate_gain("linear"))

    def forward(self, features, attn_mask, padded_summaries, teacher_forcing_ratio, max_len):
        batch_size, seq_len, features_num = features.shape
        prev_word = torch.tensor([Config.S_ID] * batch_size, device=features.device)
        hidden = (torch.zeros((1, batch_size, Config.decoder_hidden_dim), device=features.device),
                  torch.zeros((1, batch_size, Config.decoder_hidden_dim), device=features.device))

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
