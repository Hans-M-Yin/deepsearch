"""Register an existing parquet file as one DatasetRegistry split.

This is useful for an already separated evaluation set: unlike
``register_rl_dataset.py``, it does not reshuffle or split the input file.
All columns are preserved inside the workflow task's ``extra_info``.  The
source parquet is symlinked into the registry directory instead of copied;
only the Verl-wrapped parquet is materialized.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rllm.data.dataset import DatasetRegistry


def _to_builtin(value: Any) -> Any:
    """Convert Arrow/Pandas containers to values parquet can round-trip."""

    if isinstance(value, np.ndarray):
        return [_to_builtin(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register an existing parquet file without reshuffling it."
    )
    parser.add_argument("--parquet", required=True, help="Input parquet path")
    parser.add_argument("--register-name", required=True, help="Registry dataset name")
    parser.add_argument("--split", default="test", help="Registry split name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    parquet_path = Path(args.parquet).expanduser().resolve()
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)

    dataframe = pd.read_parquet(parquet_path)
    required = {"question", "answer", "images"}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    records = [
        {str(key): _to_builtin(value) for key, value in row.items()}
        for row in dataframe.to_dict("records")
    ]
    for index, record in enumerate(records):
        if not str(record.get("question", "")).strip():
            raise ValueError(f"Row {index} has an empty question")
        if record.get("answer") is None:
            raise ValueError(f"Row {index} has no answer")

    dataset_dir = Path(DatasetRegistry._DATASET_DIR) / args.register_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    registered_path = dataset_dir / f"{args.split}.parquet"
    verl_path = dataset_dir / f"{args.split}_verl.parquet"

    # Keep the original embedded-image parquet in place.  This avoids a
    # second 46MB copy on HDFS/FUSE and keeps the registry lightweight.
    if registered_path.is_symlink() or registered_path.exists():
        registered_path.unlink()
    registered_path.symlink_to(parquet_path)

    verl_data = DatasetRegistry.apply_verl_postprocessing(records)
    verl_dataframe = pd.DataFrame(verl_data)
    DatasetRegistry._write_parquet_via_tmp(verl_dataframe, str(verl_path))

    registry = DatasetRegistry._load_registry()
    registry.setdefault(args.register_name, {})[args.split] = str(registered_path)
    DatasetRegistry._save_registry(registry)

    print(f"Registered {len(records)} samples")
    print(f"name={args.register_name} split={args.split}")
    print(f"source={parquet_path}")
    print(f"data_path={registered_path}")
    print(f"verl_data_path={verl_path}")


if __name__ == "__main__":
    main()
