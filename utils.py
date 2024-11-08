from typing import List, Union
from datetime import datetime
import torch
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

# Taken from https://github.com/huggingface/transformers/blob/75f15f39a0434fe7a61385c4677f2700542a7ba6/src/transformers/data/data_collator.py#L817
def mask_inputs(
    inputs: torch.Tensor,
    vocab: dict,
    mask_prob=0.15,
    replace_prob=0.8,
    random_prob=0.1,
    special_token_border=None,
):
    """Masks inputs using the 80-10-10 strategy"""
    assert (replace_prob + random_prob) <= 1
    assert 0 <= mask_prob < 1
    # inputs must be pre-padded and a tensor
    targets = inputs.clone().long()
    probability_matrix = torch.full(targets.shape, mask_prob)
    special_tokens_mask = get_special_tokens_mask(targets, vocab, special_token_border)
 
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    targets[~masked_indices] = -100  # Only compute loss for masked indices
 
    # Replace tokens with [MASK] with probability replace_prob
    indices_replaced = (
        torch.bernoulli(torch.full(targets.shape, replace_prob)).bool() & masked_indices
    )
    inputs[indices_replaced] = vocab["[MASK]"]
 
    # Replace tokens with random with probability random_prob
    random_prob = random_prob / (
        1 - replace_prob
    )  # Adjust probability to account for already masked tokens
    indices_random = (
        torch.bernoulli(torch.full(targets.shape, random_prob)).bool()
        & masked_indices
        & ~indices_replaced
    )
    random_words = torch.randint(len(vocab), targets.shape, dtype=inputs.dtype)
    inputs[indices_random] = random_words[indices_random]
 
    # Ignore tokens with probability 1 - replace_prob - random_prob
    return inputs, targets
 
def get_special_tokens_mask(
    inputs: torch.Tensor, vocab: dict, special_token_border: int = None
):
    """Gets the special token mask for inputs"""
    if special_token_border is None:
        special_token_border = get_max_special_token_value(vocab)
    return inputs <= special_token_border
 

def get_max_special_token_value(vocab: dict):
    """Get the highest value for a special token - the special tokens must be from 0...max_value"""
    special_tokens = [v for k, v in vocab.items() if k.startswith("[")]
    assert len(special_tokens) == (
        max(special_tokens) + 1
    )  # Asserts that it is a range from 0...max_value
    return max(special_tokens)

