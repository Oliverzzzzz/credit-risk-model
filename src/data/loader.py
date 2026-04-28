import re
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parents[2] / "data"


def load_raw(filename: str = "cs-training.csv") -> pd.DataFrame:
    path = DATA_DIR / "raw" / filename
    df = pd.read_csv(path, index_col=0)
    df.columns = [_to_snake(c) for c in df.columns]
    return df


def save_processed(df: pd.DataFrame, filename: str = "processed.parquet") -> None:
    path = DATA_DIR / "processed" / filename
    df.to_parquet(path, index=False)


def load_processed(filename: str = "processed.parquet") -> pd.DataFrame:
    path = DATA_DIR / "processed" / filename
    return pd.read_parquet(path)


def _to_snake(name: str) -> str:
    s = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    return re.sub(r"[-\s]+", "_", s)
