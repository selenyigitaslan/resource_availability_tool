# resource_availability_tool
Discovering and Visualizing Resource Availability Patterns from Event Logs
<<<<<<< HEAD


Lightweight, notebook‑based toolkit to import, preprocess, and visualize event‑log data for process mining / predictive process monitoring.
Scope is intentionally focused: no PDF/HTML reports and no web dashboard — everything happens in Jupyter notebooks.

Why

The tool supports a thesis direction centered on:
How can data be preprocessed and transformed effectively for visualization?
It emphasizes practical transformations (not full feature engineering) that make event logs easier to explore and plot.

What’s inside

Notebooks (run in order):

01_import.ipynb – Load event logs (CSV/XES via pm4py), standardize columns.

02_preprocess.ipynb – Clean types, timestamps, handle missing values; transformations for plotting.

03_explore_patterns.ipynb – Activity frequencies, simple sequence/variant views (if applicable).

04_resource_profiles.ipynb – Resource workload/availability windows and overlap KPIs.

05_sensitivity.ipynb – Quick sensitivity checks on parameters (bin sizes, gaps, etc.).

Utility modules (used by the notebooks):

io_utils.py – Safe CSV loader and basic schema checks.

transform.py – Pairing start/complete events, case durations, simple time‑based enrichments.

availability.py – Build busy/active periods, daily spans, breaks, and availability heatmaps (sessionizes by gaps when lifecycle tags are missing).

multitask.py – Overlap detection & KPIs per resource (sweep‑line).

metrics.py – Lightweight timing, coverage, and sanity checks (e.g., non‑negative durations).

Expected data schema

The notebooks assume an event‑log table with (some of) the following columns:

case_id, activity, resource, timestamp, lifecycle (optional)

If lifecycle is present, use common prefixes like: start, complete, schedule, resume, suspend, abort, withdraw.
If it’s missing, availability is derived by sessionizing events using a configurable time gap.

Quick start
1) Create env (example with venv)
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

2) Install dependencies
pip install -U pip
pip install jupyterlab pandas numpy plotly kaleido pm4py packaging

3) Launch notebooks
jupyter lab  # or: jupyter notebook


Open the notebooks in order (01_… → 05_…). Point the import cell to your data file(s).

Features at a glance

Dataset overview: size, types, timestamp parsing, quick stats.

Preprocessing for visualization: missing‑data inspection, type casting, timestamp handling, useful aggregations.

Pattern exploration: activity/event frequencies, simple sequence views.

Resource profiles: active/busy intervals, day spans, idle breaks, availability per hour, and overlap KPIs.

Sensitivity checks: small parameter sweeps (e.g., bin size, gap thresholds) to see impact on plots.

Research context 
This toolkit is designed to back a literature review in two tracks:
(1) commonly used datasets in process mining/PPM, and
(2) visualization techniques and the preprocessing they require — motivating a clear, reproducible path from raw event logs to visualization‑ready tables.

Notes

No datasets are bundled; use your own (or public) logs.

Plots are built with Plotly; Kaleido enables static exports if needed.

XES support is optional via pm4py.
=======
>>>>>>> 79bdd78 (Initial commit)
