import numpy as np
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
import pyarrow.dataset as ds
from chunking import yield_chunks
import polars as pl
import pytorch_lightning as torchl
from torch.utils.data import DataLoader

class InMemoryDataset(Dataset):
    def __init__(self, dataset: ds.dataset):
        self.data, self.pnr_to_idx = self.create_db(dataset)

    def create_db(self, data):
        dataset_dict = {}
        pnr_to_idx = {}
        i = 0
        for chunk_df in yield_chunks(data, 100_000): # Unnecessary if InMemory, but kept for consistency
            for person in chunk_df.group_by("person_id").agg(pl.all().sort_by("abspos")).iter_rows(named=True):
                id = person.pop("person_id")
                dataset_dict[id] = person
                pnr_to_idx[id] = i
                i += 1
        return dataset_dict, pnr_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pnr = self.pnr_to_idx[idx]
        return self.data[pnr]


class DataModule(torchl.LightningDataModule):
    def __init__(self, sequence_path, batch_size, subset=False, padding_value=0):
        super().__init__()
        # if the padding value is negative, the token embedding fails
        assert padding_value >= 0

        self.batch_size = batch_size
        self.sequence_path = sequence_path
        self.subset = subset
        self.padding_value = padding_value
    
    def setup(self, stage=None):
        dataset = ds.dataset(self.sequence_path, format='parquet')
        if self.subset:
            dataset = dataset.take(1000)

        self.dataset = InMemoryDataset(self.dataset)
    
    def collate_fn(self, batch):

        ### Events are stored under 'events' as lists of lists
        #       where each sublist corresponds to events that occur simulatenously
        events_flat = [
                torch.tensor(event for events in item['event'] for event in events) 
                for item in batch
                ]
        # ages and abspos are stored as lists where elements corresponds to inner-level
        #   lists of events. We need to unwind this.
        ages_flat = [
            torch.tensor(
                np.repeat(item['age'], repeats=[len(events) for events in item['event']])
                for item in batch
            )
        ]
        abspos_flat = [
            torch.tensor(
                np.repeat(item['abspos'], repeats=[len(events) for events in item['event']])
                for item in batch
            )
        ]
        padded_events = pad_sequence(events_flat, batch_first=True, padding_value=self.padding_value)
        padded_age = pad_sequence(ages_flat, batch_first=True, padding_value=self.padding_value)
        padded_abspos = pad_sequence(abspos_flat, batch_first=True, padding_value=self.padding_value)
        padding_mask = torch.tensor(padded_events == self.padding_value, dtype=torch.int16)

        # placeholder implementation
        targets = torch.tensor([[1]]*len(batch))

        padded_batch = {
            'tokens': padded_events,
            'age':padded_age,
            'abspos':padded_abspos,
            'padding_mask':padding_mask,
            'targets':targets
        }

        return padded_batch

    
    def train_dataloader(self):
        # implement splitter
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        # implement splitter
        return DataLoader(self.dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        # implement splitter
        return DataLoader(self.dataset, batch_size=self.batch_size)
