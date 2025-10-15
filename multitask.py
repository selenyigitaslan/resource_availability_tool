from typing import List, Dict
import pandas as pd

def add_overlap_info(
    periods_df: pd.DataFrame,
    resource_col: str = "resource",
    start_col: str = "start",
    end_col: str = "end"
) -> pd.DataFrame:
    """
    if a new interval starts while some previous one is still open for the same resource,
    mark it as overlapping. 
    overlap_minutes captures the overlapped portion right at the start.
    """
    df = periods_df.copy()
    df[start_col] = pd.to_datetime(df[start_col], errors="coerce")
    df[end_col]   = pd.to_datetime(df[end_col], errors="coerce")
    df = df.dropna(subset=[start_col, end_col])
    df = df.sort_values([resource_col, start_col]).reset_index(drop=True)

    df["is_overlap"] = False
    df["overlap_minutes"] = 0.0

    open_ends: Dict[str, List[pd.Timestamp]] = {}

    for i, row in df.iterrows():
        res = row[resource_col]
        s   = row[start_col]
        e   = row[end_col]

        active = [x for x in open_ends.get(res, []) if x > s]
        if active:
            df.at[i, "is_overlap"] = True
            first_end = min(active)
            df.at[i, "overlap_minutes"] = max(0.0, (min(e, first_end) - s).total_seconds()/60.0)

        active.append(e)
        open_ends[res] = active

    df["overlap_minutes"] = df["overlap_minutes"].astype(float)
    return df


def overlap_summary(
    periods_df: pd.DataFrame,
    resource_col: str = "resource",
    start_col: str = "start",
    end_col: str = "end"
) -> pd.DataFrame:
    """
    Robust overlap KPI using a sweep-line per resource.

    Returns:
      [resource, n_periods, busy_min, overlap_min, overlap_share]
    where busy_min counts union minutes (active>=1), overlap_min counts minutes with active>=2.
    """
    cols = [resource_col, "n_periods", "busy_min", "overlap_min", "overlap_share"]

    if periods_df is None or periods_df.empty:
        # graceful: nothing to compute
        return pd.DataFrame(columns=cols)

    df = periods_df.copy()
    df[start_col] = pd.to_datetime(df[start_col], errors="coerce")
    df[end_col]   = pd.to_datetime(df[end_col], errors="coerce")
    df = df.dropna(subset=[start_col, end_col])
    if df.empty:
        return pd.DataFrame(columns=cols)

    df = df.sort_values([resource_col, start_col]).reset_index(drop=True)

    rows = []
    for res, g in df.groupby(resource_col):
        events = []
        for s, e in zip(g[start_col], g[end_col]):
            if e <= s:
                continue
            events.append((s, +1))
            events.append((e, -1))

        if not events:
            rows.append(dict(resource=res, n_periods=int(len(g)),
                             busy_min=0.0, overlap_min=0.0, overlap_share=0.0))
            continue

        events.sort()
        active = 0
        prev_t = events[0][0]
        busy = 0.0
        overlap = 0.0

        for t, delta in events:
            dt = (t - prev_t).total_seconds()/60.0
            if dt > 0:
                if active >= 1:
                    busy += dt
                if active >= 2:
                    overlap += dt
            active += delta
            prev_t = t

        share = (overlap / busy) if busy > 0 else 0.0
        rows.append(dict(resource=res,
                         n_periods=int(len(g)),
                         busy_min=float(busy),
                         overlap_min=float(overlap),
                         overlap_share=float(share)))

    if not rows:
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(rows)
    
    for c in cols:
        if c not in out.columns:
            out[c] = [] if c == resource_col else 0.0
    return out[cols].sort_values("overlap_share", ascending=False).reset_index(drop=True)
