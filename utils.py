from typing import List, Union
from datetime import datetime
import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.compute as pc

def get_pnrs(sources: Union[ds.Dataset, List[ds.Dataset]]) -> pa.Array:
    """Gets unique pnrs from Dataset or List of Dataset"""
    if isinstance(sources, ds.Dataset):
        return pc.unique(sources.to_table(columns=["person_id"])["person_id"])
    if isinstance(sources, list):
        return pc.unique(
            pa.concat_arrays(
                pc.unique(source.to_table(columns=["person_id"])["person_id"])
                for source in sources
            )
        )
    else:
        raise TypeError(
            f"{type(sources)} is not supported, only ds.Dataset and List[ds.Dataset]"
        )
    
def calculate_abspos(date_col: pl.Expr, origin_point=datetime(2020, 1, 1)):
    return (date_col - origin_point).dt.total_seconds() / 60 / 60

def filter_parquet_by_person_ids_to_dataset(
    file_path: str, person_ids: set[str]
) -> pa.Table:
    """
    Filters a Parquet dataset by a given set of person_ids and returns the filtered data.
 
    Args:
        file_path (str): The path to the Parquet file.
        person_ids (set[str]): A set of person IDs to filter by.
 
    Returns:
        pa.Table: A filtered PyArrow table containing rows matching the person_ids.
    """
    # Create a dataset object
    dataset = ds.dataset(file_path, format="parquet")
 
    return dataset.filter(ds.field("person_id").isin(person_ids))

