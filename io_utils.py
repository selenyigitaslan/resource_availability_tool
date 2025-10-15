from pathlib import Path
from typing import Union, List
import pandas as pd
import logging

EXPECTED_COLUMNS: List[str] = ["case_id", "activity", "resource", "timestamp", "lifecycle"]
VALID_TAG_PREFIXES = ("start", "complete", "schedule", "resume", "suspend", "abort", "withdraw")


def load_event_log_csv(
    path: Union[str, Path],
    sep: str = ",",
    expected_cols: List[str] = EXPECTED_COLUMNS,
    drop_invalid_ts: bool = True,
) -> pd.DataFrame:
    """
    Load a raw CSV
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path, sep=sep)

    # locate timestamp
    ts_candidates = [c for c in df.columns if "timestamp" in c.lower()]
    if not ts_candidates:
        raise KeyError("No column containing 'timestamp' found.")
    ts_col = ts_candidates[0]

    # required columns 
    if "resource" not in df.columns:
        df["resource"] = "Unknown"
    if "lifecycle" not in df.columns:
        df["lifecycle"] = "complete"
    for req in ["case_id", "activity"]:
        if req not in df.columns:
            raise KeyError(f"Missing required column: {req}")

    # parse timestamps 
    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True).dt.tz_convert(None)
    if drop_invalid_ts:
        n_bad = int(df[ts_col].isna().sum())
        if n_bad:
            logging.warning(f"Dropping {n_bad} rows with invalid timestamps.")
            df = df.dropna(subset=[ts_col])

    df["lifecycle"] = df["lifecycle"].astype(str).str.lower()
    df["resource"]  = df["resource"].fillna("Unknown").astype(str)
    df = df[df["lifecycle"].str.startswith(VALID_TAG_PREFIXES) | ~df["lifecycle"].notna()]

    # standardize name + order
    df = df.rename(columns={ts_col: "timestamp"})
    out_cols = [c for c in expected_cols if c in df.columns]
    return df[out_cols].sort_values("timestamp").reset_index(drop=True)
