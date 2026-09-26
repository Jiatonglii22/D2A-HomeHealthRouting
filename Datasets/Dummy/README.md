# Dummy dataset

Synthetic one-week (Mon–Fri) dataset for testing the pipeline: 6 nurses, 25 patients, 52 appointments, Pittsburgh, PA.
Regenerate with `python Datasets/generate_dummy_data.py` (seeded, so output is identical each run).

Columns marked **A** are read by the assignment model (`Optimization_Model/model.py`). The rest are for sequencing / routing and are ignored by assignment for now.

## nurse_availability.csv — one row per nurse per day

| column | used | description |
|---|---|---|
| nurse_id | A | `N001`–`N006` |
| day | A | `Mon`, `Tues`, `Wed`, `Thurs`, `Fri` (must match `days` in model.py) |
| available | A | 1 = working, 0 = off |
| start_time, end_time | A | shift, `H:MM` 24h. Blank when `available = 0`. All shifts ≤ 8 h |

## patient_appointments.csv — one row per visit

| column | used | description |
|---|---|---|
| appointment_id | | `A001`… |
| patient_id | A | `P001`–`P025`. At most one visit per patient per day |
| day | A | same labels as above |
| location | | zip code |
| lat, lon | | visit location |
| service_time | A | visit length, minutes |
| visit_type | | `SOC` start of care (90 min), `RECERT` recertification (60), `DC` discharge (45), `ROUTINE` (30/45/60) |
| required_discipline | | `RN` = RN only; `LPN` = LPN or RN |
| tw_start, tw_end | | arrival window set by the agency: visit must **start** inside it |
| priority | | 1 = must happen that day (SOC, RECERT), 2 = could move |

## travel_matrix.csv — every ordered pair of locations (992 rows)

| column | used | description |
|---|---|---|
| origin, destination | A | `Agency`, a nurse_id (that nurse's home base), or a patient_id. Assignment reads nurse → patient rows |
| travel_minutes | | drive time |
| distance_miles | A | road distance |

Distances are straight-line × 1.35 and times assume 22 mph + 3 min parking — placeholders until real road distances are pulled from a routing API.

## nurses.csv — one row per nurse

| column | description |
|---|---|
| nurse_id | |
| discipline | `RN` or `LPN` (N004, N005 are LPN) |
| home_zip, lat, lon | home base — the route starts and ends here |
| max_visits_per_day | visit cap per day (3 for the mornings-only nurse) |
| cost_per_mile | $0.50, same as `cpg` in model.py |

## patients.csv — one row per patient

| column | description |
|---|---|
| patient_id, location, lat, lon | |
| visits_per_week | 1–3 |
| window_type | `ANY` 8–17, `AM` 8–12, `PM` 12–17, `FIXED` 1-hour window |
| tw_start, tw_end | same window on every visit |
| prior_nurse_id | nurse who saw them before this week (blank = new patient, first visit is SOC) — for continuity across weeks |

## Built-in guarantees (checked by the generator)

- Every appointment has at least one qualified, on-shift nurse who can reach it inside its window.
- Each day's total visit time plus the 30-minute buffer fits within total nurse hours. The same holds for RN-only visits against RN hours, and visit counts fit within the visit caps.
- A greedy router can schedule every day with time windows, skills and visit caps, so a feasible full routing exists.
