from typing import List
import pandas as pd


def extract_start_end(
    event_log: pd.DataFrame,
    case_id_col: str = "case_id",
    resource_col: str = "resource",
    activity_col: str = "activity",
    timestamp_col: str = "timestamp",
    lifecycle_col: str = "lifecycle",
    start_label: str = "START",
    end_label: str = "COMPLETE",
) -> pd.DataFrame:
    """
    Pair START -> COMPLETE using FIFO.
    """
    df = event_log.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
    df = df.dropna(subset=[timestamp_col])

    start_p = str(start_label).lower()
    end_p   = str(end_label).lower()

    rows: List[dict] = []
    key_cols = [case_id_col, resource_col, activity_col]

    # FIFO
    for keys, g in df.sort_values(key_cols + [timestamp_col]).groupby(key_cols):
        queue: List[pd.Timestamp] = []  # unmatched starts
        for _, r in g.iterrows():
            tag = str(r[lifecycle_col]).lower()
            t = r[timestamp_col]
            if tag.startswith(start_p):
                queue.append(t)
            elif tag.startswith(end_p) and queue:
                s = queue.pop(0)
                if pd.isna(s) or pd.isna(t) or t < s:
                    continue
                rows.append({
                    case_id_col:  keys[0],
                    resource_col: keys[1],
                    activity_col: keys[2],
                    "start_time": s,
                    "end_time":   t,
                    "duration":   t - s,
                    "duration_seconds": float((t - s).total_seconds()),
                })

    return pd.DataFrame(rows).reset_index(drop=True)


def compute_case_duration(
    event_log: pd.DataFrame,
    case_id_col: str = "case_id",
    first_timestamp: str = "start_time",
    second_timestamp: str = "end_time",
    out_column: str = "case_duration",
) -> pd.DataFrame:
    """Per-case duration = [min start, max end]"""
    df = event_log.copy()
    df[first_timestamp]  = pd.to_datetime(df[first_timestamp],  errors="coerce")
    df[second_timestamp] = pd.to_datetime(df[second_timestamp], errors="coerce")

    g = df.groupby(case_id_col).agg(
        case_start=(first_timestamp, "min"),
        case_end  =(second_timestamp, "max"),
    ).reset_index()

    g[out_column] = g["case_end"] - g["case_start"]
    g[f"{out_column}_seconds"] = g[out_column].dt.total_seconds()
    return g


def add_day_of_week(
    df: pd.DataFrame,
    timestamp_col: str = "start_time",
    out_col: str = "day_of_week",
) -> pd.DataFrame:
    """Append weekday (0=Mon..6=Sun) from a timestamp column."""
    out = df.copy()
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
    out[out_col] = out[timestamp_col].dt.weekday
    return out


def add_seconds_in_day(
    df: pd.DataFrame,
    timestamp_col: str = "start_time",
    out_col: str = "seconds_in_day",
) -> pd.DataFrame:
    """Append seconds since midnight from a timestamp column."""
    out = df.copy()
    ts = pd.to_datetime(out[timestamp_col], errors="coerce")
    out[out_col] = ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second
    return out
