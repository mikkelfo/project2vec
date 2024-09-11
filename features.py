""" File for creating various features such as age, abspos, segment, etc."""
 
from typing import List
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
import polars.selectors as cs
 
from tokenize import tokenize # type: ignore
from chunking import yield_chunks
from utils import calculate_abspos
 

def create_background(df: pl.DataFrame) -> pl.DataFrame:
    """Creates background dataframe - to be used as a source"""
    # Rename birthday to date_col
    df = df.rename({"birthday": "date_col"})
    # Convert all background_cols to background column
    df = df.melt(
        id_vars=["person_id", "date_col"],
        value_name="background_col",
    )
    return df.select("person_id", "date_col", "background_col")
 

def add_cls_token(lf: pl.LazyFrame, birthdates: pl.LazyFrame, cls_token: int = 1):
    """Adds cls tokens to the start of the first event"""
    # Create CLS_token and reorder to lf order
    cls_lf = (
        birthdates.with_columns(event=pl.lit([cls_token]))
        .rename({"birthday": "date_col"})
        .collect()
        .lazy()
    )  # Concat doesnt properly without - consider finding "proper" fix
    return pl.concat((cls_lf, lf), rechunk=True)
 

def add_sep_tokens(lf: pl.LazyFrame, sep_token: int = 2):
    """Adds seperator tokens to the end of each (TOKENIZED) event"""
    return lf.with_columns(pl.col("event").list.concat(sep_token))
 

def create_abspos(lf: pl.LazyFrame, origin_point=datetime(2020, 1, 1)):
    """Creates the abspos columns by doing 'date_col' - origin_point"""
    return lf.with_columns(
        calculate_abspos(pl.col("date_col"), origin_point=origin_point).alias("abspos")
    )
 

def create_age(lf: pl.LazyFrame, birthdates: pl.LazyFrame):
    """Creates the age column by joining birthdates and subtracting from date_col"""
    return (
        lf.join(birthdates, on="person_id", how="inner")
        .with_columns(
            ((pl.col("date_col") - pl.col("birthday")).dt.total_days() / 365.25).alias(
                "age"
            )
        )
        .drop("birthday")
    )
 

def create_segment(lf: pl.LazyFrame):
    """Creates the segment column by doing cumcount"""
    return lf.with_columns(
        (pl.col("person_id").cum_count().over("person_id")).alias("segment") - 1
    )
 

def create_tokenized_events(sources: List[pl.LazyFrame], vocab: dict, dir_name: Path):
    """
    Tokenizes and creates events, saving it to dir_name / 'tokenized_parquet'
 
    Parameters:
        sources: The list of LazyFrames to be processed and tokenized
        vocab: The vocabulary used for tokenization
        dir_name: The "name" of this data version
    Assumptions:
        Sources to have ID='person_id' and timestamp='date_col'
    """
    file_path = dir_name / "tokenized.parquet"
    file_path.parent.mkdir(parents=True, exist_ok=True)
 
    schema = pa.schema(
        [
            pa.field("person_id", pa.int64()),
            pa.field("date_col", pa.timestamp("ns")),
            pa.field("event", pa.large_list(pa.int64())),
        ]
    )
    writer = pq.ParquetWriter(file_path, schema=schema)
 
    for chunk_dfs in tqdm(yield_chunks(sources)):
        combined_df = pl.DataFrame(
            schema={
                "person_id": pl.Int64,
                "date_col": pl.Datetime("ns"),
                "event": pl.List(pl.Int64),
            }
        )
        for chunk_df in chunk_dfs:
            # Tokenize string columns
            chunk_df = tokenize(chunk_df, vocab)
 
            # Select event columns (by negation of non-event columns)
            event_columns = ~cs.by_name("person_id", "date_col")
 
            # Convert to dataframe of (person_id, date_col, event)
            chunk_df = (
                chunk_df.with_columns(
                    pl.Series("event", chunk_df.select(event_columns).to_numpy())
                )
                .select("person_id", "date_col", "event")  # Only select needed columns
                .cast({"event": pl.List(pl.Int64)})  # Convert numpy NaNs to Null
            )
            # Drop Nulls from event list
            chunk_df = chunk_df.with_columns(pl.col("event").list.drop_nulls())
 
            combined_df = pl.concat([combined_df, chunk_df], how="vertical_relaxed")
        combined_df = combined_df.sort("date_col")
        writer.write_table(combined_df.to_arrow())
    writer.close()
 
    return pl.scan_parquet(file_path)