"""Dataset loading helpers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import config as C


def load_dataset(path: Path | str = C.RAW_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.generate_dataset")
    df = pd.read_csv(path, encoding="utf-8")
    missing = [c for c in C.ALL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    df["Size"] = df["Size"].astype(int)
    df["Price"] = df["Price"].astype(int)
    df["Trial_Timestamp"] = pd.to_datetime(df["Trial_Timestamp"])
    return df


def load_or_generate() -> pd.DataFrame:
    """Load the CSV, generating it first (fixed seed) if it does not exist yet."""
    if not C.RAW_PATH.exists():
        from src import generate_dataset
        df = generate_dataset.generate()
        checks = generate_dataset.validate(df)
        C.RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(C.RAW_PATH, index=False, encoding="utf-8")
        generate_dataset.write_documents(df, checks)
    return load_dataset()
