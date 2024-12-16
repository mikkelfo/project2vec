import torch
import torch.nn as nn
from transformer.transformer_utils import ReZero

import logging
log = logging.getLogger(__name__)


class Embeddings(nn.Module):
    """Class for token, position, segment and backgound embedding."""

    def __init__(self, hparams):
        super(Embeddings, self).__init__()
        embedding_size = hparams.hidden_size

        # Initialize Token/Concept embedding matrix
        self.token = nn.Embedding(
            hparams.vocab_size,
            embedding_size,
            padding_idx=0
        )

        # Initialize Time2Vec embeddings
        self.age = PositionalEmbedding(1, hparams.hidden_size, torch.cos)
        self.abspos = PositionalEmbedding(1, hparams.hidden_size, torch.sin)
        # Uniformly initialise the weights of the embedding matrix
        d = 0.01
        nn.init.uniform_(self.token.weight, a=-d, b=d)

        self.res_age = ReZero(hparams.hidden_size, simple=True, fill=0)
        self.res_abs = ReZero(hparams.hidden_size, simple=True, fill=0)
        self.dropout = nn.Dropout(hparams.emb_dropout)

        # Initialize the partner type embedding
        self.partner_type = nn.Embedding(3, embedding_size, padding_idx=0)

    def forward(self, tokens, position, age, partner_type=None):
        """"""
        tokens = self.token(tokens)

        pos = self.age(age.float().unsqueeze(-1))
        tokens = self.res_age(tokens, pos)
        pos = self.abspos(position.float().unsqueeze(-1))
        tokens = self.res_abs(tokens, pos)

        if partner_type is not None:
            partner_type = self.partner_type(partner_type)
            tokens = tokens + partner_type

        return self.dropout(tokens), None


# TIME2VEC IMPLEMENTATION

def t2v(tau, f, w, b, w0, b0, arg=None):
    """Time2Vec function"""
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], -1)


class PositionalEmbedding(nn.Module):
    """Implementation of Time2Vec"""

    def __init__(self, in_features, out_features, f):
        super(PositionalEmbedding, self).__init__()

        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.w = nn.parameter.Parameter(
            torch.randn(in_features, out_features - 1))
        self.b = nn.parameter.Parameter(
            torch.randn(in_features, out_features - 1))
        self.f = f

        d = 0.01
        nn.init.uniform_(self.w0, a=-d, b=d)
        nn.init.uniform_(self.b0, a=-d, b=d)
        nn.init.uniform_(self.w, a=-d, b=d)
        nn.init.uniform_(self.b, a=-d, b=d)

    def forward(self, tau):
        return t2v(tau, self.f, self.w, self.b, self.w0, self.b0)
