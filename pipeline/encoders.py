"""Part 2 sentence encoders and our MNLI pair wrapper.

Source: nlp1-student/part2/clslib.py (EmbedAndPoolEncoder and LSTMEncoder),
also demonstrated in part2/tutorial.ipynb. These sentence encoders retain the
course computation and constructor arguments; docstrings/comments are shortened.
PairEncoder is our addition for premise/hypothesis inputs, marked inline below.

Course license: licenses/NLP1-COURSE-LICENSE (Apache-2.0).
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class EmbedAndPoolEncoder(nn.Module):
    # like tutorial

    def __init__(self, embedding_matrix, average_pool=True, padding_idx=0, freeze_embeddings=True):
        super().__init__()
        self.average_pool = average_pool
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=freeze_embeddings,
            padding_idx=padding_idx,
        )

    @property
    def enc_dim(self):
        return self.embedding.embedding_dim

    @property
    def pad_idx(self):
        return self.embedding.padding_idx

    def forward(self, x):
        #Map IDs van shape (batch_size, max_seq_length) naar 1 vector
        e = self.embedding(x)
        if self.average_pool: #
            den = torch.sum(x.ne(self.embedding.padding_idx), dim=-1, keepdim=True).clamp_min(1)
        else:
            den = 1
        h = torch.sum(e, -2) / den
        return h


class LSTMEncoder(nn.Module):
    # Tutorial
    def __init__(self, hidden_size, embedding_matrix, padding_idx=0, freeze_embeddings=True, word_dropout=0.):
        # Tutorial single layer, unidirectional LSTM en word dropout
        super().__init__()
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=freeze_embeddings,
            padding_idx=padding_idx,
        )
        self.hidden_size = hidden_size
        self.word_dropout = word_dropout
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

    @property
    def enc_dim(self):
        return self.hidden_size

    @property
    def pad_idx(self):
        return self.embedding.padding_idx

    def forward(self, x):
        embedded = self.embedding(x)
        valid = x.ne(self.embedding.padding_idx)

        if self.training and self.word_dropout > 0:
            drop_mask = (
                torch.rand(x.shape, device=embedded.device) < self.word_dropout
            ) & valid
            embedded = embedded.masked_fill(drop_mask.unsqueeze(-1), 0.0)

        lengths = torch.sum(valid, dim=-1)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, (hidden, cell) = self.lstm(packed)
        final_hidden = hidden[-1]
        return final_hidden


class PairEncoder(nn.Module):  # MNLI addition: wrap either tutorial sentence encoder.
    # encode premise en hypothese los met dezelfde encoder, combineer daarna.
    # features="concat":    [p; h]                      (2 x enc_dim)
    # features="infersent": [p; h; |p-h|; p*h]          (4 x enc_dim), Conneau et al. 2017 (D-015)

    def __init__(self, sentence_encoder, features="concat"):
        super().__init__()
        if features not in ("concat", "infersent"):
            raise ValueError(f"Unknown pair features: {features}")
        self.sentence_encoder = sentence_encoder  # share weights van beide zinnen.
        self.features = features

    @property
    def enc_dim(self):
        factor = 2 if self.features == "concat" else 4
        return factor * self.sentence_encoder.enc_dim  # classifier input size follows the feature choice

    @property
    def pad_idx(self):
        return self.sentence_encoder.pad_idx  # MNLI: both sentences use the same vocabulary.

    def forward(self, x):
        premise, hypothesis = x  #twee padded token-ID batches.
        p = self.sentence_encoder(premise)
        h = self.sentence_encoder(hypothesis)
        if self.features == "concat":
            return torch.cat([p, h], dim=-1)
        return torch.cat([p, h, torch.abs(p - h), p * h], dim=-1)  # explicit difference + agreement per dimension
