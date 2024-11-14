import torch
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl

# Custom dataset to generate synthetic data


class SyntheticDataset(Dataset):
    def __init__(self, num_samples, vocab_size, max_length):
        self.num_samples = num_samples
        self.vocab_size = vocab_size
        self.max_length = max_length

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        tokens = torch.randint(0, self.vocab_size, (self.max_length,))
        abspos = torch.arange(0, self.max_length)
        age = torch.randint(0, 100, (1,)).repeat(self.max_length)
        padding_mask = torch.zeros(self.max_length)
        targets = torch.randint(0, 2, (1,)).float()
        return {
            "tokens": tokens,
            "abspos": abspos,
            "age": age,
            "padding_mask": padding_mask,
            "targets": targets,
        }

# PyTorch Lightning DataModule


class SyntheticDataModule(pl.LightningDataModule):
    def __init__(self, vocab_size, max_length, batch_size, num_samples):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_samples = num_samples

    def setup(self, stage=None):
        self.dataset = SyntheticDataset(
            self.num_samples, self.vocab_size, self.max_length)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
