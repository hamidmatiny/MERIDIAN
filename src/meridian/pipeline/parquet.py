"""Deterministic Parquet IO for generated frames."""

from pathlib import Path
from typing import cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write ``frame`` so the same values produce the same bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    pq.write_table(
        table,
        path,
        compression="snappy",
        use_dictionary=False,
        write_statistics=False,
    )


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a frame written by :func:`write_parquet`."""
    return cast("pd.DataFrame", pq.read_table(path).to_pandas())
