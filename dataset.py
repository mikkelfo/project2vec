import json
from torch.utils.data import Dataset
import pyarrow.dataset as ds
from chunking import yield_chunks
import polars as pl
import pytorch_lightning as torchl
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from collate_fn import CensorCollate, PartnerCensorCollate

import pandas as pd

class InMemoryDataset(Dataset):
    def __init__(self, dataset: ds.dataset, training_targets: dict, holdout_targets: dict, prop_val):
        self.holdout_indices = set()
        self.data = self.create_db(dataset, training_targets, holdout_targets)
        self.pnrs = list(self.data.keys())
        self.non_holdout_indices, self.holdout_indices = self.split_by_target(training_targets, holdout_targets)
        self.train_indices, self.val_indices = self.split_data(prop_val=prop_val)

    def create_db(self, data, training_targets: dict, holdout_targets: dict):
        dataset_dict = {} # mapping from person_id to data point
        i = 0
        for chunk_df in yield_chunks(data, 100_000): # Unnecessary if InMemory, but kept for consistency
            for person in chunk_df.group_by("person_id").agg(pl.all().sort_by("abspos")).iter_rows(named=True):
                id = person.pop("person_id")

                # check if the person is in the training or holdout/training set
                training_obs = training_targets.get(id, False)
                holdout_obs = holdout_targets.get(id, False)

                # check if we have target for person
                if training_obs or holdout_obs:
                    dataset_dict[id] = person
                    if holdout_obs:
                        dataset_dict[id]['target'] = holdout_targets[id]
                    else:
                        dataset_dict[id]['target'] = training_targets[id]
                    i += 1

        return dataset_dict

    def __len__(self):
        return len(self.pnrs)

    def __getitem__(self, idx):
        pnr = self.pnrs[idx]
        return {key: val for key, val in self.data[pnr].items()}
    
    def split_by_target(self, training_targets, holdout_targets):
        """
        Separate indices into training and holdout based on the provided dictionaries.
        """

        training_indices = [i for i, pnr in enumerate(self.pnrs) if pnr in training_targets]
        holdout_indices = [i for i, pnr in enumerate(self.pnrs) if pnr in holdout_targets]

        return training_indices, holdout_indices

    def split_data(self, prop_val):
        """ 
        Split the training observations further into training and validation sets
        """

        train_indices, val_indices = train_test_split(self.non_holdout_indices, test_size=prop_val, random_state=42)

        return train_indices, val_indices

    def get_training_data(self):
        """
        Return the training data.
        """
        train_data = {self.pnrs[idx]: self.data[self.pnrs[idx]] for idx in self.train_indices}
        return InMemoryDatasetSubset(train_data)

    def get_validation_data(self):
        """
        Return validation data.
        """
        val_data = {self.pnrs[idx]: self.data[self.pnrs[idx]] for idx in self.val_indices}
        return InMemoryDatasetSubset(val_data)

    def get_testing_data(self):
        """
        Return the holdout data.
        """
        test_data = {self.pnrs[idx]: self.data[self.pnrs[idx]] for idx in self.holdout_indices}
        return InMemoryDatasetSubset(test_data)

    
class InMemoryDatasetSubset(InMemoryDataset):
    def __init__(self, data):
        self.data = data
        self.pnrs = list(self.data.keys())


class PartnerInMemoryDataset(InMemoryDataset):
    def __init__(self, dataset: ds.dataset, targets: dict, partners: dict, prop_val):
        self.data = self.create_db(dataset, targets, partners)
        self.pnrs = list(self.data.keys())
        self.train_indices, self.val_indices = self.split_data(prop_val=prop_val)

    def create_db(self, data, targets: dict, partners: dict):
        dataset_dict = {} # mapping from person_id to data point
        i = 0
        for chunk_df in yield_chunks(data, 100_000): # Unnecessary if InMemory, but kept for consistency
            for person in chunk_df.group_by("person_id").agg(pl.all().sort_by("abspos")).iter_rows(named=True):
                id = person.pop("person_id")
                if id in targets:
                    dataset_dict[id] = person
                    dataset_dict[id]['target'] = targets[id]
                    dataset_dict[id]['partner'] = partners[id]
                    i += 1

        return dataset_dict

    def __getitem__(self, idx):
        pnr = self.pnrs[idx]
        person = {key: val for key, val in self.data[pnr].items()}
        partner_id = person.pop('partner')
        if partner_id in self.data:
            partner = {key: val for key, val in self.data[partner_id].items() if key not in ["partner", "target"]}
        else:
            partner = {}
        return person, partner

    def get_training_data(self):
        train_data = {i: self.data[i] for i in self.train_indices}
        return ParentInMemoryDatasetSubset(train_data)

    def get_validation_data(self):
        val_data = {i: self.data[i] for i in self.val_indices}
        return ParentInMemoryDatasetSubset(val_data)

class ParentInMemoryDatasetSubset(PartnerInMemoryDataset):
    def __init__(self, data):
        self.data = data
        self.pnrs = list(self.data.keys())


class DataModule(torchl.LightningDataModule):
    def __init__(self, sequence_path, batch_size, target_path, holdout_target_path, vocab_path, subset=False, prop_val=0.2):
        super().__init__()
        
        self.batch_size = batch_size
        self.sequence_path = sequence_path
        self.target_path = target_path
        self.holdout_target_path = holdout_target_path
        self.subset = subset
        self.prop_val = prop_val
        self.train_indices = None
        self.val_indices = None

        # check vocab is correct (only needed for synthetic data)
        with open (vocab_path, 'r') as f:
            vocab = json.load(f)
        assert vocab["[PAD]"] == 0


        self.collate_fn = CensorCollate(
            truncate_length=512, # TODO: Adjust accordingly
            background_length=0, # TODO: Need proper value
            segment=False,
            negative_censor=0,
        )
    
    def setup(self, stage=None):
        dataset = ds.dataset(self.sequence_path, format='parquet')
        targets = pd.read_csv(self.target_path)
        holdout_targets = pd.read_csv(self.holdout_target_path)

        if self.subset:
            targets = targets.head(1000)
        
        training_targets = targets.set_index('person_id')['target'].squeeze().to_dict()
        holdout_targets = holdout_targets.set_index('person_id')['target'].squeeze().to_dict()

        self.dataset = InMemoryDataset(
            dataset,
            training_targets=training_targets,
            holdout_targets=holdout_targets,
            prop_val=0.2
            )
    
    def train_dataloader(self):
        # split dataset into train and val
        return DataLoader(self.dataset.get_training_data(), batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn)
    
    def val_dataloader(self):
        return DataLoader(self.dataset.get_validation_data(), batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)

    def test_dataloader(self):
        # implement splitter
        return DataLoader(self.dataset.get_testing_data(), batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn)


class PartnerDataModule(DataModule):
    def __init__(self, sequence_path, batch_size, target_path, vocab_path, subset=False, prop_val=0.2):
        super().__init__(sequence_path, batch_size, target_path, vocab_path, subset, prop_val)

        self.collate_fn = PartnerCensorCollate(
            truncate_length=512, # TODO: Adjust accordingly
            background_length=0, # TODO: Need proper value
            segment=False,
            negative_censor=0,
        )

    def setup(self, stage=None):
        dataset = ds.dataset(self.sequence_path, format='parquet')
        targets = pd.read_csv(self.target_path)

        if self.subset:
            targets = targets.head(1000)

        targets_dict = targets.set_index('person_id')['target'].squeeze().to_dict()
        partners_dict = targets.set_index('person_id')['partner'].squeeze().to_dict()
        self.dataset = PartnerInMemoryDataset(dataset, targets=targets_dict, partners=partners_dict, prop_val=0.2)

