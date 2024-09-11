""" File for creating the vocabulary and tokenizing the dataframes """
 
from typing import List
from collections import Counter
import polars as pl
import polars.selectors as cs
 

def create_vocab(sources: List[pl.LazyFrame], cutoff=0):
    """Creates vocab from String columns of sources"""
    vocab = {
        "[PAD]": 0,
        "[CLS]": 1,
        "[SEP]": 2,
        "[UNK]": 3,
        "[MASK]": 4,
    }
    counts = Counter()
    for source_df in sources:
        # Convert to single column, drop nulls and get count of each unique value
        source_count = (
            source_df.select(cs.string())
            .melt()
            .drop_nulls()
            .group_by("value")
            .len()
            .collect(streaming=True)
        )
        # Convert to dictionary and add to counts
        source_count = dict(zip(*source_count))
        counts += Counter(source_count)
 
    for key, value in counts.items():
        if value >= cutoff and key not in vocab:
            vocab[key] = len(vocab)
 
    return vocab
 

def tokenize(df: pl.DataFrame, vocab: dict):
    """Tokenize all String columns of the dataframe"""
    hack_vocab = (
        vocab.copy()
    )  # Copy just to make sure nothing goes wrong down the line with vocab
    hack_vocab[pl.Null] = None
    return df.with_columns(
        cs.string().replace(
            hack_vocab, default=hack_vocab["[UNK]"], return_dtype=pl.Int64
        )
    ) # type: ignore