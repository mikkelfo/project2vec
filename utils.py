from datetime import datetime
import polars as pl
from typing import List, Union

def get_pnrs(df: Union[pl.LazyFrame, List[pl.LazyFrame]]) -> pl.Series:
    """Returns all 'person_ids' of LazyFrame or List of LazyFrames"""
    if isinstance(df, pl.LazyFrame):
        return df.select("person_id").unique().collect()
 
    elif isinstance(df, pl.DataFrame):
        return df.select("person_id").unique()
    elif isinstance(df, list):
        return pl.concat(df).select("person_id").unique().collect()
    else:
        raise TypeError(
            f"{type(df)} is not supported, only pl.LazyFrame, pl.DataFrame and List[pl.LazyFrame]"
        )
    
def calculate_abspos(date_col: pl.Expr, origin_point=datetime(2020, 1, 1)):
    return (date_col - origin_point).dt.total_seconds() / 60 / 60