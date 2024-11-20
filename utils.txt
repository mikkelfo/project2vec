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

