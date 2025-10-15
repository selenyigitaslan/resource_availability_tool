from typing import List, Tuple, Union
import pandas as pd

def get_active_periods(
    df: pd.DataFrame,
    start_tags: Tuple[str, ...] = ("start", "schedule", "resume"),
    end_tags: Tuple[str, ...]   = ("complete", "suspend", "abort", "withdraw"),
    tolerance: Union[str, pd.Timedelta] = "5min",
    default_event_dur: Union[str, pd.Timedelta] = "5min",
) -> pd.DataFrame:
    """
    Build the *union* of busy intervals per resource.

    - If lifecycle tags exist we use a counter per resource.
    - If not, we sessionize by gaps: events <= tolerance apart are merged.
    - Afterward, we merge tiny gaps <= tolerance.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values(["resource", "timestamp"])

    tol = pd.to_timedelta(tolerance)
    def_dur = pd.to_timedelta(default_event_dur)

    has_lifecycle = "lifecycle" in df.columns
    tags = df["lifecycle"].astype(str).str.lower() if has_lifecycle else None
    has_start = (tags.str.startswith(start_tags).any() if has_lifecycle else False)
    has_end   = (tags.str.startswith(end_tags).any()   if has_lifecycle else False)

    records: List[dict] = []

    if has_lifecycle and has_start and has_end:
        # Counter-based union
        for res, g in df.assign(tag=tags).groupby("resource"):
            active = 0
            cur_start = None
            for ts, tag in zip(g["timestamp"], g["tag"]):
                if any(tag.startswith(p) for p in start_tags):
                    if active == 0:
                        cur_start = ts
                    active += 1
                elif any(tag.startswith(p) for p in end_tags):
                    if active > 0:
                        active -= 1
                        if active == 0 and cur_start is not None:
                            records.append({"resource": res, "start": cur_start, "end": ts})
                            cur_start = None
    else:
        # No lifecycle => Sessionize by time gaps per resource
        for res, g in df.groupby("resource"):
            g = g.sort_values("timestamp")
            session_start, prev_ts = None, None
            for ts in g["timestamp"]:
                if session_start is None:
                    session_start = ts
                elif prev_ts is not None and (ts - prev_ts) > tol:
                    records.append({"resource": res, "start": session_start, "end": prev_ts + def_dur})
                    session_start = ts
                prev_ts = ts
            if session_start is not None and prev_ts is not None:
                records.append({"resource": res, "start": session_start, "end": prev_ts + def_dur})

    periods = pd.DataFrame.from_records(records, columns=["resource", "start", "end"])
    if periods.empty:
        return periods

    periods = periods.sort_values(["resource", "start"]).reset_index(drop=True)

    # Merge tiny gaps (<= tolerance)
    merged = []
    for res, g in periods.groupby("resource"):
        g = g.sort_values("start").reset_index(drop=True)
        cur_s, cur_e = g.loc[0, "start"], g.loc[0, "end"]
        for _, r in g.iloc[1:].iterrows():
            if r["start"] - cur_e <= tol:
                cur_e = max(cur_e, r["end"])
            else:
                merged.append({"resource": res, "start": cur_s, "end": cur_e})
                cur_s, cur_e = r["start"], r["end"]
        merged.append({"resource": res, "start": cur_s, "end": cur_e})
    return pd.DataFrame(merged)


def compute_daily_spans(periods: pd.DataFrame) -> pd.DataFrame:
    """
    Per resource & day: first_start, last_end, busy_min, span_min, pct_busy
    """
    p = periods.copy()
    p["start"] = pd.to_datetime(p["start"])
    p["end"]   = pd.to_datetime(p["end"])
    p["day"]   = p["start"].dt.date

    busy = (
        p.assign(dur=(p["end"] - p["start"]).dt.total_seconds() / 60)
         .groupby(["resource","day"])["dur"].sum()
         .reset_index(name="busy_min")
    )
    span = (
        p.groupby(["resource","day"])
         .agg(first_start=("start","min"), last_end=("end","max"))
         .reset_index()
    )

    out = span.merge(busy, on=["resource","day"], how="left")
    out["span_min"] = (out["last_end"] - out["first_start"]).dt.total_seconds() / 60
    out["pct_busy"] = (out["busy_min"] / out["span_min"]).clip(lower=0, upper=1)
    return out


def extract_breaks(periods: pd.DataFrame) -> pd.DataFrame:
    """
    Intra-day idle gaps between consecutive union periods (same resource, same day).
    """
    p = periods.copy()
    p["start"] = pd.to_datetime(p["start"])
    p["end"]   = pd.to_datetime(p["end"])
    p["day"]   = p["start"].dt.date

    rows: List[dict] = []
    for (res, day), g in p.sort_values(["resource","start"]).groupby(["resource","day"]):
        prev_end = None
        for _, r in g.iterrows():
            if prev_end is not None and r["start"] > prev_end:
                gap = (r["start"] - prev_end).total_seconds() / 60
                rows.append({
                    "resource": res, "day": day,
                    "gap_min": gap, "gap_start": prev_end, "gap_end": r["start"],
                    "start_hour": prev_end.hour
                })
            prev_end = r["end"] if prev_end is None else max(prev_end, r["end"])
    return pd.DataFrame(rows)


def weekly_availability_matrix(periods: pd.DataFrame) -> pd.DataFrame:
    """
    Resource × hour availability: share of days with any activity in that hour.
    Output: long form [resource, hour, availability_share].
    """
    p = periods.copy()
    p["start"] = pd.to_datetime(p["start"])
    p["end"]   = pd.to_datetime(p["end"])
    p["day"]   = p["start"].dt.date

    days_per_res = p.groupby("resource")["day"].nunique().rename("n_days").reset_index()

    def hours_touched(s: pd.Timestamp, e: pd.Timestamp) -> list[int]:
        start_h = s.floor("h")
        end_h   = (e - pd.Timedelta(seconds=1)).floor("h")
        if end_h < start_h:
            return [int(s.hour)]
        n = int((end_h - start_h) / pd.Timedelta(hours=1)) + 1
        return [int((start_h + i * pd.Timedelta(hours=1)).hour) for i in range(n)]

    tall = []
    for (res, day), g in p.groupby(["resource", "day"]):
        hours = set()
        for _, r in g.iterrows():
            for h in hours_touched(r["start"], r["end"]):
                hours.add(h)
        for h in hours:
            tall.append({"resource": res, "day": day, "hour": h})

    tall = pd.DataFrame(tall)
    if tall.empty:
        return pd.DataFrame(columns=["resource", "hour", "availability_share"])

    n_days_with_hour = (
        tall.groupby(["resource", "hour"])["day"]
            .nunique()
            .rename("n_days_with_hour")
            .reset_index()
    )

    out = n_days_with_hour.merge(days_per_res, on="resource", how="left")
    out["availability_share"] = out["n_days_with_hour"] / out["n_days"]
    return out[["resource", "hour", "availability_share"]]
