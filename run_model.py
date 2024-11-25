# General
import argparse
import logging


# Model and Data
from models.classifier import TransformerEncoder
from pytorch_lightning import Trainer
from utils import get_device

import torch


log = logging.getLogger(__name__)


def main(args):
    pass


if __name__ == "__main__":
    """
    Run the training
    """

    parser = argparse.ArgumentParser(description="Run the model training.")

    parser.add_argument("--hidden_size", type=int, default=48,
                        help="Size of the hidden layers and embeddings.")
    parser.add_argument("--hidden_ff", type=int, default=96,
                        help="Size of the position-wise feed-forward layer.")
    parser.add_argument("--n_encoders", type=int, default=4,
                        help="Number of encoder blocks.")
    parser.add_argument("--n_heads", type=int, default=2,
                        help="Number of attention heads in the multiheadattention module.")
    parser.add_argument("--n_local", type=int, default=2,
                        help="Number of local attention heads.")
    parser.add_argument("--local_window_size", type=int, default=4,
                        help="Size of the window for local attention.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size for training.")
    parser.add_argument("--max_length", type=int, default=30,
                        help="Maximum length of the input sequence.")
    parser.add_argument("--vocab_size", type=int, default=100,
                        help="Size of the vocabulary.")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate for optimization.")
    parser.add_argument("--num_epochs", type=int, default=30,
                        help="Number of epochs for training.")
    parser.add_argument("--attention_type", type=str, default="performer",
                        help="Type of attention mechanism to use.")
    parser.add_argument("--norm_type", type=str, default="rezero",
                        help="Normalization type to use.")
    parser.add_argument("--num_random_features", type=int, default=32,
                        help="Number of random features for the Attention module (Performer uses this).")
    parser.add_argument("--emb_dropout", type=float, default=0.1,
                        help="Dropout rate for the embedding block.")
    parser.add_argument("--fw_dropout", type=float, default=0.1,
                        help="Dropout rate for the position-wise feed-forward layer.")
    parser.add_argument("--att_dropout", type=float, default=0.1,
                        help="Dropout rate for the multiheadattention module.")
    parser.add_argument("--dc_dropout", type=float, default=0.1,
                        help="Dropout rate for the decoder block.")
    parser.add_argument("--hidden_act", type=str, default="swish",
                        help="Activation function for the hidden layers (attention layers use ReLU).")
    parser.add_argument("--epsilon", type=float, default=1e-8,
                        help="Epsilon value for numerical stability.")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                        help="Weight decay for optimizer.")
    parser.add_argument("--beta1", type=float, default=0.9,
                        help="Beta1 hyperparameter for Adam optimizer.")
    parser.add_argument("--beta2", type=float, default=0.999,
                        help="Beta2 hyperparameter for Adam optimizer.")

    args = parser.parse_args()
