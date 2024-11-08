from pathlib import Path
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

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