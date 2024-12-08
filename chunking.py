from pathlib import Path
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pyarrow.compute as pc
import polars as pl
from utils import get_pnrs

def yield_chunks(dataset: ds.Dataset, chunk_size: int) -> ds.Dataset:
    pnrs = get_pnrs(dataset)
    for i in range(0, dataset.count_rows(), chunk_size):
        chunk_pnrs = pnrs[i : i + chunk_size]
        yield pl.from_arrow(dataset.to_table(filter=pc.is_in(pc.field("person_id"), chunk_pnrs)))

def write_dataset_to_parquet_in_batches(
    dataset: ds.Dataset, output_path: Path, batch_size: int = 10_000_000
) -> ds.Dataset:
    """
    Write a PyArrow Dataset to Parquet in batches and return a new Dataset pointing to the output path.
    Faster and less memory than ds.write_dataset()
 
    Args:
        dataset (ds.Dataset): PyArrow Dataset to write.
        output_path (Path): Path to write the Parquet file.
        batch_size (int): Number of rows per batch.
 
    Returns:
        ds.Dataset: A new PyArrow Dataset pointing to the output path.
    """
    with pq.ParquetWriter(output_path, dataset.schema) as writer:
        for batch in dataset.to_batches(batch_size=batch_size):
            writer.write_table(pa.Table.from_batches([batch]))
 
    return ds.dataset(output_path, format="parquet")