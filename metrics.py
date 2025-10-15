# metrics.py
# -----------------------------------------------------------------------------
# Utility functions for evaluation: coverage, runtimes, applicability,
# and automatic sanity checks for availability & multitasking outputs.
# Includes stability/regularity helpers provided by the user.
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Iterable

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Small file/IO helpers
# -----------------------------------------------------------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _append_csv(df: pd.DataFrame, out_path: str) -> None:
    """Append df to CSV, write header only if file does not exist."""
    _ensure_dir(os.path.dirname(out_path) or ".")
    header = not os.path.exists(out_path)
    df.to_csv(out_path, mode="a", header=header, index=False)


def _now_str() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def build_dataset_tag(dataset_name: str, cfg: Optional[Dict] = None) -> str:
    """Create a readable tag for sensitivity runs: adds bin size / smoothing if provided."""
    if not cfg:
        return dataset_name
    parts = [dataset_name]
    if "bin_size_minutes" in cfg:
        parts.append(f"bin{cfg['bin_size_minutes']}")
    if "min_gap_minutes" in cfg:
        parts.append(f"gap{cfg['min_gap_minutes']}")
    if "overlap_mode" in cfg:
        parts.append(str(cfg["overlap_mode"]))
    return "_".join(map(str, parts))


# -----------------------------------------------------------------------------
# RUNTIME TIMER
# -----------------------------------------------------------------------------

@contextmanager
def stage_timer(stage_name: str):
    """Context manager to measure wall-clock time of a stage."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t0
        print(f"[metrics] {stage_name} took {elapsed:.3f}s")
        # let caller record this with record_runtimes


# -----------------------------------------------------------------------------
# COVERAGE / RUNTIMES / APPLICABILITY LOGGING
# -----------------------------------------------------------------------------

def record_coverage(dataset_name: str,
                    views: Dict[str, bool],
                    artifacts_dir: str = "artifacts",
                    cfg: Optional[Dict] = None) -> None:
    """Log which predefined views were successfully produced."""
    tag = build_dataset_tag(dataset_name, cfg)
    row = {"dataset": tag, "ts": _now_str(), **views}
    _append_csv(pd.DataFrame([row]), os.path.join(artifacts_dir, "eval_coverage.csv"))


def record_runtimes(dataset_name: str,
                    stage_times: Dict[str, float],
                    artifacts_dir: str = "artifacts",
                    cfg: Optional[Dict] = None) -> None:
    """Log wall-clock time per stage (seconds)."""
    tag = build_dataset_tag(dataset_name, cfg)
    row = {"dataset": tag, "ts": _now_str(), **stage_times}
    _append_csv(pd.DataFrame([row]), os.path.join(artifacts_dir, "eval_runtimes.csv"))


def record_applicability(dataset_name: str,
                         applicability: Dict[str, float],
                         artifacts_dir: str = "artifacts",
                         cfg: Optional[Dict] = None) -> None:
    """Log which transformations were applied and their counts."""
    tag = build_dataset_tag(dataset_name, cfg)
    row = {"dataset": tag, "ts": _now_str(), **applicability}
    _append_csv(pd.DataFrame([row]), os.path.join(artifacts_dir, "eval_applicability.csv"))


# -----------------------------------------------------------------------------
# SANITY CHECKS
# -----------------------------------------------------------------------------

@dataclass
class SanityResult:
    dataset: str
    check: str
    passed: bool
    violations: int
    severity: str   # "info" | "warn" | "error"
    notes: str


def _union_minutes(intervals: Iterable[tuple[pd.Timestamp, pd.Timestamp]]) -> int:
    """Union length in minutes for a list of (start,end) timestamps."""
    iv = sorted(intervals, key=lambda x: x[0])
    if not iv:
        return 0
    merged = [iv[0]]
    for s, e in iv[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    delta = sum(int((e - s).total_seconds() // 60) for s, e in merged)
    return delta


def _normalize_overlap_columns(overlap_bins: pd.DataFrame) -> pd.DataFrame:
    """Allow either 'coalesced_min' / 'prop_min' or
       'workload_coalesced_min' / 'workload_proportional_min'."""
    df = overlap_bins.copy()
    if "coalesced_min" not in df.columns and "workload_coalesced_min" in df.columns:
        df = df.rename(columns={"workload_coalesced_min": "coalesced_min"})
    if "prop_min" not in df.columns and "workload_proportional_min" in df.columns:
        df = df.rename(columns={"workload_proportional_min": "prop_min"})
    return df


def run_sanity_checks(
    events_std: pd.DataFrame,
    availability_bins: Optional[pd.DataFrame] = None,
    overlap_bins: Optional[pd.DataFrame] = None,
    cfg: Optional[Dict] = None,
    dataset_name: str = "dataset"
) -> pd.DataFrame:
    """
    Execute automatic sanity checks S1..S4 (+S5 candidate) and append to artifacts/sanity_report.csv.
    Returns the dataframe of results for convenience.
    """
    out: List[SanityResult] = []
    eps = 1e-6
    bin_size = int((cfg or {}).get("bin_size_minutes", 60))

    # S1: schema & timestamps monotonic per case (approximate check on sort)
    required = {"case_id", "activity", "resource", "timestamp"}
    missing = required - set(map(str, events_std.columns))
    s1_pass = (len(missing) == 0)
    out.append(SanityResult(dataset_name, "S1_required_columns", s1_pass, len(missing), "error",
                            f"missing={sorted(missing)}"))
    # optional monotonic: per case, check any negative diff
    if "case_id" in events_std.columns and "timestamp" in events_std.columns:
        tmp = events_std[["case_id", "timestamp"]].copy()
        tmp["timestamp"] = pd.to_datetime(tmp["timestamp"], errors="coerce")
        tmp = tmp.dropna(subset=["timestamp"]).sort_values(["case_id", "timestamp"])
        tmp["diff"] = tmp.groupby("case_id")["timestamp"].diff().dt.total_seconds()
        neg = int((tmp["diff"] < -eps).sum())
        out.append(SanityResult(dataset_name, "S1_time_non_decreasing", neg == 0, neg, "warn",
                                "negative time jumps within cases"))

    # S2: durations/waits >= 0 where present
    dur_neg = int(((events_std.get("duration_sec", pd.Series(dtype=float)) < 0)).sum()) \
        if "duration_sec" in events_std else 0
    wait_neg = int(((events_std.get("wait_sec", pd.Series(dtype=float)) < 0)).sum()) \
        if "wait_sec" in events_std else 0
    out.append(SanityResult(dataset_name, "S2_non_negative", (dur_neg + wait_neg) == 0,
                            dur_neg + wait_neg, "error", ""))

    # S3: availability semantics
    if availability_bins is not None and not availability_bins.empty:
        ab = availability_bins.copy()
        if "available_ratio" in ab.columns:
            ratio_oob = int(((ab["available_ratio"] < -eps) | (ab["available_ratio"] > 1 + eps)).sum())
            out.append(SanityResult(dataset_name, "S3_ratio_in_[0,1]", ratio_oob == 0, ratio_oob, "error", ""))
        if {"available", "available_ratio"}.issubset(ab.columns):
            flag_mismatch = int((ab["available"].astype(bool) != (ab["available_ratio"] > 0)).sum())
            out.append(SanityResult(dataset_name, "S3_flag_consistency", flag_mismatch == 0, flag_mismatch, "error", ""))

    # S4: overlap invariants per bin
    if overlap_bins is not None and not overlap_bins.empty:
        ob = _normalize_overlap_columns(overlap_bins)
        if "coalesced_min" in ob.columns:
            coalesced_oob = int((ob["coalesced_min"] > bin_size + eps).sum())
            out.append(SanityResult(dataset_name, "S4_coalesced_le_bin", coalesced_oob == 0, coalesced_oob, "error", ""))
        if {"prop_min", "coalesced_min"}.issubset(ob.columns):
            prop_vs_union = int((np.abs(ob["prop_min"] - ob["coalesced_min"]) > 1.0).sum())  # 1 min tolerance
            out.append(SanityResult(dataset_name, "S4_prop_vs_union_close", prop_vs_union == 0, prop_vs_union, "warn",
                                    "tolerance=1 minute"))
        if availability_bins is not None and not availability_bins.empty:
            merged = ob.merge(availability_bins[["resource", "bin_start", "available"]],
                              on=["resource", "bin_start"], how="left")
            bad = int(((merged["available"] == True) & (merged.get("concurrent_count", 0) < 1)).sum())
            out.append(SanityResult(dataset_name, "S4_available_implies_concurrency>=1", bad == 0, bad, "warn", ""))

    # Return as DataFrame
    df = pd.DataFrame([asdict(r) for r in out])
    return df


def save_sanity_report(results: pd.DataFrame,
                       artifacts_dir: str = "artifacts",
                       cfg: Optional[Dict] = None,
                       dataset_name: str = "dataset") -> None:
    if results.empty:
        return
    tag = build_dataset_tag(dataset_name, cfg)
    results = results.copy()
    results["dataset"] = tag
    results["ts"] = _now_str()
    _append_csv(results, os.path.join(artifacts_dir, "sanity_report.csv"))


# -----------------------------------------------------------------------------
# APPLICABILITY-INFERENCE (optional if transform doesn't return counts)
# -----------------------------------------------------------------------------

def infer_applicability_from_events_std(events_std: pd.DataFrame) -> Dict[str, float]:
    """
    Heuristically infer which transformations were applied using columns/flags
    that may be present in events_std. If a column is missing, return NaN for that metric.
    Expected optional columns:
      - duration_heuristic (bool): duration was imputed
      - unfolded (bool): interval created from start/complete
      - is_duplicate (bool): duplicate rows flagged during cleaning
    """
    out: Dict[str, float] = {}
    out["n_events"] = float(len(events_std))
    for col, key in [("duration_heuristic", "n_heuristic"),
                     ("unfolded", "n_unfolded"),
                     ("is_duplicate", "n_dupes_marked")]:
        if col in events_std.columns:
            out[key] = float(events_std[col].astype(bool).sum())
        else:
            out[key] = float("nan")
    # convenience rates
    if out["n_events"] and not np.isnan(out["n_events"]):
        for key in ["n_heuristic", "n_unfolded", "n_dupes_marked"]:
            if not np.isnan(out[key]):
                out[key.replace("n_", "rate_")] = out[key] / out["n_events"]
            else:
                out[key.replace("n_", "rate_")] = float("nan")
    return out


# -----------------------------------------------------------------------------
# STABILITY / REGULARITY HELPERS 
# -----------------------------------------------------------------------------

def stability_score(weekly_long: pd.DataFrame, resource: str) -> float:
    """
    Cosine similarity between first-half and second-half hourly availability
    vector within a week for the given resource (0..1).
    Expects columns: ['resource','hour','availability_share'] with hour in [0..23].
    """
    w = weekly_long[weekly_long["resource"] == resource].copy()
    if w.empty:
        return 0.0
    w = w.sort_values(["hour"]).reset_index(drop=True)
    w["half"] = (w.index % 2).astype(int)
    a = (
        w[w["half"] == 0]
        .set_index("hour")["availability_share"]
        .reindex(range(24), fill_value=0)
        .values
    )
    b = (
        w[w["half"] == 1]
        .set_index("hour")["availability_share"]
        .reindex(range(24), fill_value=0)
        .values
    )
    if np.all(a == 0) or np.all(b == 0):
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def stability_score_by_day(periods: pd.DataFrame, resource: str) -> float:
    """Cosine similarity between first-half and second-half daily 24h activity vectors (0..1)."""
    p = periods[periods["resource"] == resource].copy()
    if p.empty:
        return 0.0

    p["start"] = pd.to_datetime(p["start"])
    p["end"] = pd.to_datetime(p["end"])
    p = p.dropna(subset=["start", "end"])
    p = p[p["end"] > p["start"]]
    p["day"] = p["start"].dt.date

    days = sorted(p["day"].unique())
    if len(days) < 2:
        return 0.0

    def vec24(df: pd.DataFrame) -> np.ndarray:
        v = pd.Series(0.0, index=range(24))
        for _, r in df.iterrows():
            sh = r["start"].floor("h")
            eh = (r["end"] - pd.Timedelta(seconds=1)).floor("h")
            cur = sh
            while cur <= eh:
                v[int(cur.hour)] = 1.0
                cur += pd.Timedelta(hours=1)
        return v.values

    mid = len(days) // 2
    v1 = vec24(p[p["day"].isin(days[:mid])])
    v2 = vec24(p[p["day"].isin(days[mid:])])

    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def regularity_index(daily_spans: pd.DataFrame, resource: str) -> dict:
    """Arrival/leave regularity -> lower = more regular.
    Expects columns: ['resource','first_start','last_end'] (datetime)."""
    d = daily_spans[daily_spans["resource"] == resource].copy()
    if d.empty:
        return {"cv_start": np.nan, "cv_end": np.nan}
    s = d["first_start"].dt.hour * 60 + d["first_start"].dt.minute
    e = d["last_end"].dt.hour * 60 + d["last_end"].dt.minute
    return {
        "cv_start": float(np.std(s) / (np.mean(s) + 1e-9)),
        "cv_end": float(np.std(e) / (np.mean(e) + 1e-9)),
    }


def lunch_presence_rate(breaks: pd.DataFrame, resource: str, min_minutes: int = 20) -> float:
    """% of days with a gap ≥ min_minutes between 11:00–14:00.
    Expects columns: ['resource','day','start_hour','gap_min']."""
    b = breaks[breaks["resource"] == resource]
    if b.empty:
        return 0.0
    total = b["day"].nunique()
    lunch = b[(b["start_hour"].between(11, 14)) & (b["gap_min"] >= min_minutes)]["day"].nunique()
    return 0.0 if total == 0 else float(lunch / total)
