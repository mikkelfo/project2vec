import json
import numpy as np
from torch.utils.data import Dataset
import torch
from torch.nn.utils.rnn import pad_sequence
import pyarrow.dataset as ds
from chunking import yield_chunks
import polars as pl
import pytorch_lightning as torchl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import pandas as pd

class InMemoryDatasetSubset(Dataset):
    """ 
    Dataset class to for train and validation subsets
    of the full dataset class below. 
    """
    def __init__(self, data, idx_to_pnr):
        self.data = data
        self.idx_to_pnr = idx_to_pnr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pnr = self.idx_to_pnr[idx]
        return self.data[pnr]


class InMemoryDataset(Dataset):
    def __init__(self, dataset: ds.dataset, targets: dict, prop_val):
        self.data, self.idx_to_pnr= self.create_db(dataset, targets)
        self.train_indices, self.val_indices = self.split_data(prop_val=prop_val)

    def create_db(self, data, targets: dict):
        dataset_dict = {} # mapping from person_id to data point
        idx_to_pnr= {} # mapping from dataset index to person_id
        i = 0
        for chunk_df in yield_chunks(data, 100_000): # Unnecessary if InMemory, but kept for consistency
            for person in chunk_df.group_by("person_id").agg(pl.all().sort_by("abspos")).iter_rows(named=True):
                id = person.pop("person_id")
                if id in targets:
                    dataset_dict[id] = person
                    dataset_dict[id]['target'] = targets[id]
                    idx_to_pnr[i] = id
                    i += 1

        return dataset_dict, idx_to_pnr

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pnr = self.idx_to_pnr[idx]
        return self.data[pnr]

    def split_data(self, prop_val):
        indices = list(self.idx_to_pnr.keys())
        train_indices, val_indices = train_test_split(indices, test_size=prop_val, random_state=42)
        return train_indices, val_indices

    def get_training_data(self):
        train_data = {i: self.data[i] for i in self.train_indices}
        train_idx_to_pnr = {i: self.idx_to_pnr[i] for i in self.train_indices}
        return InMemoryDatasetSubset(train_data, train_idx_to_pnr)

    def get_validation_data(self):
        val_data = {i: self.data[i] for i in self.val_indices}
        val_idx_to_pnr = {i: self.idx_to_pnr[i] for i in self.val_indices}
        return InMemoryDatasetSubset(val_data, val_idx_to_pnr)
  

class DataModule(torchl.LightningDataModule):
    def __init__(self, sequence_path, batch_size, target_path, vocab_path, subset=False, prop_val=0.2):
        super().__init__()
        
        self.batch_size = batch_size
        self.sequence_path = sequence_path
        self.target_path = target_path
        self.subset = subset
        self.prop_val = prop_val
        self.train_indices = None
        self.val_indices = None

        # set padding value to be the next integer after the largest key in the vocab
        with open (vocab_path, 'r') as f:
            vocab = json.load(f)

        max_key = max(vocab.values())
        self.padding_value = max_key + 1
    
    def setup(self, stage=None):
        dataset = ds.dataset(self.sequence_path, format='parquet')
        targets = pd.read_csv(self.target_path)

        if self.subset:
            targets = targets.head(1000)
        
        targets = targets.set_index('person_id')['target'].squeeze().to_dict()
        self.dataset = InMemoryDataset(dataset, targets=targets, prop_val=0.2)
    
    def collate_fn(self, batch):
        # Flatten `event`, `age`, and `abspos` fields for each item in the batch
        events_flat = [
            torch.tensor([event for events in item['event'] for event in events])
            for item in batch
        ]
        ages_flat = [
            torch.tensor(
                np.repeat(item['age'], repeats=[len(events) for events in item['event']])
            )
            for item in batch
        ]
        abspos_flat = [
            torch.tensor(
                np.repeat(item['abspos'], repeats=[len(events) for events in item['event']])
            )
            for item in batch
        ]

        # Pad sequences to the maximum length in the batch
        padded_events = pad_sequence(events_flat, batch_first=True, padding_value=self.padding_value)
        padded_age = pad_sequence(ages_flat, batch_first=True, padding_value=0)
        padded_abspos = pad_sequence(abspos_flat, batch_first=True, padding_value=0)

        # Create a padding mask
        padding_mask = (padded_events == self.padding_value).to(torch.int64)

        # Collect targets
        targets = torch.tensor([[item['target']] for item in batch], dtype=torch.float64)

        # Construct the padded batch dictionary
        padded_batch = {
            'tokens': padded_events,
            'age': padded_age,
            'abspos': padded_abspos,
            'padding_mask': padding_mask,
            'targets': targets,
        }

        return padded_batch
    
    
    def train_dataloader(self):
        # split dataset into train and val
        return DataLoader(self.dataset.get_training_data(), batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.dataset.get_validation_data(), batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)

    def test_dataloader(self):
        # implement splitter
        #return DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)
        return None
