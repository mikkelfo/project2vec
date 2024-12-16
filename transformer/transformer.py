import torch
import torch.nn as nn
from transformer.embeddings import Embeddings
from transformer.transformer_utils import gelu, gelu_new, swish, ScaleNorm
from transformer.modules import EncoderLayer
import logging
from typing import Dict

log = logging.getLogger(__name__)

ACT2FN = {
    "gelu": torch.nn.functional.gelu,
    "gelu_custom": gelu,
    "relu": torch.nn.functional.relu,
    "swish": swish,
    "gelu_google": gelu_new,
    "tanh": torch.tanh,
}


class Transformer(nn.Module):
    def __init__(self, hparams):
        """Encoder part of the life2vec model"""
        super(Transformer, self).__init__()

        self.hparams = hparams
        # Initialize the Embedding Layer
        self.embedding = Embeddings(hparams=hparams)
        # Initialize the Encoder Blocks
        self.encoders = nn.ModuleList(
            [EncoderLayer(hparams) for _ in range(hparams.n_encoders)]
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass
        Input:
            batch: dict with keys: event, position, age, segment, padding_mask
        """
        x, _ = self.embedding(
            tokens=batch["event"],
            position=batch["abspos"],
            age=batch["age"],
            partner_type=batch.get("partner_type"),
        )
        for layer in self.encoders:
            x = torch.einsum("bsh, bs -> bsh", x, batch["padding_mask"])
            x = layer(x, mask=batch["padding_mask"])
        return x

    def redraw_projection_matrix(self, batch_idx: int):
        """Redraw projection Matrices for each layer (only valid for Performer)"""
        if batch_idx == -1:
            log.info("Redrawing projections for the encoder layers (manually)")
            for encoder in self.encoders:
                encoder.redraw_projection_matrix()

        elif batch_idx > 0 and batch_idx % self.hparams.feature_redraw_interval == 0:
            log.info("Redrawing projections for the encoder layers")
            for encoder in self.encoders:
                encoder.redraw_projection_matrix()


class CLS_Decoder(nn.Module):
    """Classification based on the CLS token"""

    def __init__(self, hparams):
        super(CLS_Decoder, self).__init__()
        hidden_size = hparams.hidden_size
        p = hparams.dc_dropout

        self.in_layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=p)
        self.norm = ScaleNorm(hidden_size=hidden_size, eps=hparams.epsilon)

        self.act = ACT2FN["swish"]
        self.out_layer = nn.Linear(hidden_size, 1)

    def forward(self, x, **kwargs):
        """Foraward Pass"""
        x = x[:, 0]
        x = self.dropout(self.norm(self.act(self.in_layer(x))))
        return self.out_layer(x)
