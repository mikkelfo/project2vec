# General
import argparse
import logging


# Model and Data
from models.classifier import TransformerEncoder
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from dataloaders.synthetic import SyntheticDataModule


log = logging.getLogger(__name__)


def main(args):
    log.warning("(INFO) Starting the training process.")
    model = TransformerEncoder(args.__dict__)
    dataloader = SyntheticDataModule(num_samples=1000, max_length=args.max_length,
                                     batch_size=args.batch_size, vocab_size=args.vocab_size)

    model_checkpoint = ModelCheckpoint(
        monitor='val/ap', save_top_k=2, save_last=True, mode='max')
    early_stopping = EarlyStopping(monitor='val/ap', patience=5, mode='max')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    logger = TensorBoardLogger("lightning_logs", name="cls_logs")

    trainer = Trainer(max_epochs=30,
                      accelerator=args.device,  # change to "cuda" or "gpu" or 'msp'
                      limit_train_batches=0.5,
                      logger=logger,
                      accumulate_grad_batches=4,
                      num_sanity_val_steps=8,
                      callbacks=[model_checkpoint, early_stopping, lr_monitor],
                      check_val_every_n_epoch=1)

    trainer.fit(model, dataloader)


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
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to use for training.")

    args = parser.parse_args()
    main(args)
