"""Write SpaceTravLR-compatible betadata Feather (Arrow IPC + LZ4)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather


def write_betadata_feather(
    path: Path,
    cell_ids: list[str],
    beta_columns: list[str],
    betas: np.ndarray,
    id_col: str = "CellID",
) -> None:
    """betas shape (n_cells, n_betas) aligned with beta_columns."""
    if betas.shape[0] != len(cell_ids):
        raise ValueError("row count mismatch")
    if betas.shape[1] != len(beta_columns):
        raise ValueError("column count mismatch")

    data = {id_col: cell_ids}
    for j, col in enumerate(beta_columns):
        data[col] = betas[:, j].astype(np.float64)
    df = pd.DataFrame(data)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    feather.write_feather(table, path, compression="lz4")
