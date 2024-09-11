from typing import List, Union
import polars as pl
from utils import get_pnrs

def yield_chunks(lf: Union[pl.LazyFrame, List[pl.LazyFrame]], chunk_size=100_000):
    """ Gets all unique person_ids from LazyFrame or List of LazyFrames and yields chunks of chunk_size """

    # Get all person_ids
    pnrs = get_pnrs(lf)

    # Next, iterate over them and return each source filtered by chunks
    for i in range(0, len(pnrs), chunk_size):
        chunk_pnrs = pnrs[i : i + chunk_size].lazy()

        if isinstance(lf, pl.LazyFrame):
            yield lf.join(chunk_pnrs, on="person_id", how="inner").collect(streaming=True)
        elif isinstance(lf, list):
            yield [df.join(chunk_pnrs, on="person_id", how="inner").collect(streaming=True) for df in lf]