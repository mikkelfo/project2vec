import torch
import torch.nn as nn
import torchmetrics as metrics
import pytorch_lightning as pl
import logging

"""Custom code"""
from transformer.transformer import Transformer, CLS_Decoder

log = logging.getLogger(__name__)


class TransformerEncoder(pl.LightningModule):
    """Transformer with Classification Task"""

    def __init__(self, hparams):
        super(TransformerEncoder, self).__init__()
        self.hparams.update(hparams)

        self.last_global_step = 0
        # 1. ENCODER
        self.transformer = Transformer(self.hparams)
        # 2. DECODER BLOCK
        self.decoder = CLS_Decoder(self.hparams)
        # 3. LOSS
        self.loss = nn.BCEWithLogitsLoss()
        # 4. METRICS
        self.init_metrics()

    def init_metrics(self):
        self.train_loss = metrics.MeanMetric()
        self.val_loss = metrics.MeanMetric()
        self.test_loss = metrics.MeanMetric()

        self.train_acc = metrics.Accuracy(task="binary")
        self.val_acc = metrics.Accuracy(task="binary")
        self.test_acc = metrics.Accuracy(task="binary")

        self.train_mcc = metrics.MatthewsCorrCoef(task="binary")
        self.val_mcc = metrics.MatthewsCorrCoef(task="binary")
        self.test_mcc = metrics.MatthewsCorrCoef(task="binary")

        self.train_ap = metrics.AveragePrecision(task="binary")
        self.val_ap = metrics.AveragePrecision(task="binary")
        self.test_ap = metrics.AveragePrecision(task="binary")

    def log_metrics(self, predictions, targets,  loss, stage: str):
        """Log the metrics"""
        predictions = predictions.detach()
        targets = targets.detach()
        if stage == "train":
            self.train_loss(loss.detach())
            self.train_acc(predictions, targets)
            self.train_mcc(predictions, targets)
            self.train_ap(predictions, targets.long())

            self.log("train/loss", self.train_loss,
                     on_step=True, on_epoch=True)
            self.log("train/acc", self.train_acc, on_step=True, on_epoch=True)
            self.log("train/mcc", self.train_mcc, on_step=True, on_epoch=True)
            self.log("train/ap", self.train_ap, on_step=True, on_epoch=True)

        elif stage == "val":
            self.val_loss(loss.detach())
            self.val_acc(predictions, targets)
            self.val_mcc(predictions, targets)
            self.val_ap(predictions, targets.long())

            self.log("val/loss", self.val_loss, on_step=True, on_epoch=True)
            self.log("val/acc", self.val_acc, on_step=True, on_epoch=True)
            self.log("val/mcc", self.val_mcc, on_step=True, on_epoch=True)
            self.log("val/ap", self.val_ap, on_step=True, on_epoch=True)

        elif stage == "test":
            self.test_loss(loss.detach())
            self.test_acc(predictions, targets)
            self.test_mcc(predictions, targets)
            self.test_ap(predictions, targets.long())

            self.log("test/loss", self.test_loss, on_step=True, on_epoch=True)
            self.log("test/acc", self.test_acc, on_step=True, on_epoch=True)
            self.log("test/mcc", self.test_mcc, on_step=True, on_epoch=True)
            self.log("test/ap", self.test_ap, on_step=True, on_epoch=True)

        else:
            raise NotImplementedError()
        return None

    def print_metrics(self, loss, acc, mcc, stage: str):
        print(
            f'\n{stage.capitalize()} Metrics\n'
            f'\tLoss: {loss:.3f}\n'
            f'\tAccuracy: {acc:.3f}\n'
            f'\tMCC: {mcc:.3f}\n'
        )

    def forward(self, batch):
        """Forward pass that returns the logits for the masked language model and the sequence order prediction task."""
        # 1. ENCODER INPUT
        encoded = self.transformer(batch)
        # 2. MASKED LANGUAGE MODEL
        logits = self.decoder(encoded)
        preds = nn.functional.sigmoid(logits)

        return {"logits": logits,
                "preds": preds}

    def steps_per_epoch(self):
        num_batches = self.trainer.limit_train_batches
        if num_batches == None:
            num_batches = 100
        accumulate_grad_batches = self.trainer.accumulate_grad_batches
        return (
            num_batches + accumulate_grad_batches - 1) // accumulate_grad_batches

    def training_step(self, batch, batch_idx):
        """Training Step"""
        # 1. ENCODER-DECODER
        output = self(batch)
        # 2. LOSS
        loss = self.loss(output["logits"], batch["targets"])
        # 3. LOGGING
        self.log_metrics(
            output["preds"], batch["targets"], loss, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation Step"""
        # 1. ENCODER-DECODER
        output = self(batch)
        # 2. LOSS
        loss = self.loss(output["logits"], batch["targets"])
        # 3. LOGGING
        self.log_metrics(
            output["preds"], batch["targets"], loss, stage="val")
        return loss

    def test_step(self, batch, batch_idx):
        # 1. ENCODER-DECODER
        output = self(batch)
        # 2. LOSS
        loss = self.loss(output["logits"], batch["targets"])
        # 3. LOGGING
        self.log_metrics(
            output["preds"], batch["targets"], loss, stage="test")
        return loss

    def on_train_epoch_end(self, *kwargs):
        """On Train Epoch End: Redraw the projection of the Attention-related matrices"""
        if self.hparams.attention_type == "performer":
            self.transformer.redraw_projection_matrix(-1)
        else:
            raise NotImplementedError(
                "We only have a Performer implementation.")

        # loss, acc, mcc = self.train_loss.compute(
        # ), self.train_acc.compute(), self.train_mcc.compute()
        # self.log("train_loss",  loss)
        # self.log("train_acc", acc)
        # self.log("train_mcc", mcc)
        # self.print_metrics(loss, acc, mcc, stage="train")

    # def on_validation_epoch_end(self, *kwargs):
    #     loss, acc, mcc = self.val_loss.compute(
    #     ), self.val_acc.compute(), self.val_mcc.compute()
    #     self.log("val_loss",  loss, )
    #     self.log("val_acc", acc)
    #     self.log("val_mcc", mcc)
    #     self.print_metrics(loss, acc, mcc, stage="val")

    # def on_test_epoch_end(self, *kwargs):
    #     loss, acc, mcc = self.test_loss.compute(
    #     ), self.test_acc.compute(), self.test_mcc.compute()
    #     self.log("test_loss",  loss)
    #     self.log("test_acc", acc)
    #     self.log("test_mcc", mcc)
    #     self.print_metrics(loss, acc, mcc, stage="test")

    def configure_optimizers(self):
        """Configuration of the Optimizer and the Learning Rate Scheduler."""
        no_decay = [
            "bias",
            "norm",
            "age",
            "abspos",
            "token",
            "decoder.g"
        ]

        # It is advised to avoid the decay on the embedding weights, biases of the model and values of the ReZero gates.

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            betas=(self.hparams.beta1, self.hparams.beta2),
            eps=self.hparams.epsilon,
        )

        if self.steps_per_epoch() == 0:
            steps_per_epoch = 100
        else:
            steps_per_epoch = self.steps_per_epoch()

        log.warning(f"Steps per epoch: {steps_per_epoch}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                    optimizer, max_lr=self.hparams.learning_rate,
                    epochs=self.hparams.num_epochs, steps_per_epoch=steps_per_epoch,
                    three_phase=False, pct_start=0.05, max_momentum=self.hparams.beta1,
                    div_factor=30
                ),
                "interval": "step",
                "frequency": 1,
                "name": "learning_rate",
            },
        }
