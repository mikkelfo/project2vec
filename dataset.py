from torch.utils.data import Dataset
import pyarrow.dataset as ds
from chunking import yield_chunks
import polars as pl

class InMemoryDataset(Dataset):
    def __init__(self, dataset: ds.dataset):
        self.data, self.pnr_to_idx = self.create_db(dataset)

    def create_db(self, data):
        dataset_dict = {}
        pnr_to_idx = {}
        i = 0
        for chunk_df in yield_chunks(data, 100_000): # Unnecessary if InMemory, but kept for consistency
            for person in chunk_df.groupby("person_id").agg(pl.all().sort_by("abspos")).iter_rows(named=True):
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
