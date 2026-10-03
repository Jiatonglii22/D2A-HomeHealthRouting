# D2A-HomeHealthRouting

## Prediction Model

Agency-level demand allocation model is used to estimate agency-specific demand spatially using ACS need indicators, then use that estimated demand to create synthetic patients that Gurobi will route.

We're using UPMC Home Health Agency in Greater Pittsburgh Region as the agency we are allocating demand for in its serviced zip codes.

## Setup

Python 3.10 or newer (tested on 3.12). Assignment needs a Gurobi license for models over 2,000 variables; the current forecast week is about 6,600.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pipeline

The optimization runs in two layers:

1. **Assignment (Layer 1)**, `Optimization_Model/model.py`, run by `main.py`: a Gurobi MILP that decides which nurse sees which patient each day. It minimizes mileage cost from each nurse's home plus a continuity penalty `w_c` for every extra nurse a patient sees.
2. **Sequencing (Layer 2)**, `Optimization_Model/sequencing.py`: takes those assignments and finds the visit order and timetable for every nurse-day.

```bash
python main.py                            # Layer 1 -> Outputs/nurse_daily_assignments.csv, patient_continuity.csv
python -m Optimization_Model.sequencing   # Layer 2 -> Outputs/route_*.csv, unroutable_nurse_days.csv
```

## Sequencing (Layer 2)

Each nurse-day has only a handful of visits, so instead of a routing MILP, sequencing tries every possible visit order and keeps the cheapest one by miles that the nurse can actually drive.

| Choice | What we do | Why |
|---|---|---|
| Method | Try every order, up to 8 visits per nurse-day | Exact, not a heuristic. 8 visits is 40,320 orders, under a second. |
| Above 8 visits | Nearest neighbour, then 2-opt | 9 visits is 362,880 orders. Routes from it are marked `method = heuristic`. |
| Route start and end | Each nurse's home | Matches the cost basis assignment uses. `depot='Agency'` routes from the agency instead, for comparison. |
| Arriving early | Wait for the window to open, and record the wait | A visit can't start early. Wait time is a real cost of narrow windows. |
| Objective | Fewest miles | Consistent with assignment's mileage cost. |

Every visit order is checked three ways: early arrivals wait, a visit must start before its window closes, and the nurse must be home by the end of her shift. An order that fails any check is discarded, so a chosen route is always one a nurse could drive.

Visit length and time windows are looked up by (patient, day), because the same patient can need different visit lengths on different days. The forecast has no time windows, so every visit is open for the whole shift and wait time is 0. If `tw_start`/`tw_end` columns are added to the appointments file, they are used automatically.

### Inputs

| File | Used for |
|---|---|
| `Outputs/nurse_daily_assignments.csv` | Layer 1's assignments (which nurse sees whom, each day) |
| `Datasets/Processed/synthetic_appointments_397636.csv` | visit length per (patient, day) |
| `Datasets/Processed/travel_car.csv` | travel minutes and miles between every pair of locations (agency, nurse homes, patients) |
| `Datasets/Forecast/nurse_availability.csv` | shifts. These are the same shifts Layer 1 solves with; there is no shift data in Processed. |

### Outputs (in `Outputs/`)

| File | Contents |
|---|---|
| `route_schedule.csv` | one row per stop: arrive, start and end times, wait, and drive time and miles from the previous stop |
| `route_schedule_<day>.csv` | the same, one file per day |
| `route_summary.csv` | one row per nurse-day: route, miles, cost, drive and wait minutes, return time, minutes to spare before shift end, and method |
| `unroutable_nurse_days.csv` | nurse-days with no order that fits, with the patients and the reason |

### Current results (forecast week, 11 nurses, `w_c = 0.5`)

| Measure | Value |
|---|---|
| Nurse-days routed | 48 of 50, all by trying every order |
| Visits scheduled | 190 of 203 |
| Route miles / cost | 1,499.3 mi / $352.33 at $0.235 per mile |
| Unroutable nurse-days | N008 Mon (shift overrun by 7 min), N006 Wed (overrun by 1 min) |
| Run time | under 1 second |

The code was checked against a brute-force search over every visit order: on all 48 routed nurse-days, the chosen route is the cheapest feasible one.

## Limitations

**Assignment underestimates driving time, so a few nurse-days can't be routed.** Assignment's capacity constraint (`hours_worked`) budgets the one-way drive from the nurse's home to each patient, but not the drive back home at the end of the day. Layer 2 drives the full route, home → patients → home. So assignment underestimates daily driving on 37 of 50 nurse-days, by 27% on average. The gap is largest on light days: with one patient 40 minutes away, assignment budgets 40 minutes of driving, while the real route is 80.

The effect on the current week is small: 2 of 50 nurse-days overrun the shift by 1 and 7 minutes (13 of 203 visits), and N010 Thu finishes with 0.1 minutes to spare. Layer 2 reports these rather than hiding them, in `unroutable_nurse_days.csv`.

This limitation will grow with fewer nurses. In the planned staffing sweep, rosters of about 7–9 nurses make days fuller and tighter, so expect more unroutable nurse-days. Decide on the fix below before running the sweep.

**Proposed fix (not yet applied; up to the assignment model's owner):** add one variable per nurse-day, `home[i,d] >= tau[j][i] * x[i,j,d]` for every patient j, and add `+ home[i,d]` to the left side of `hours_worked`. This reserves at least the drive home from the farthest patient the nurse visits. It stays linear and leaves the objective unchanged. Also read `travel_minutes` as `float` rather than `int` (model.py line 60), which currently drops the fraction of every travel time.

Tested end to end on a copy of the model:

- Assignment still solves to optimal (objective 360.21 → 366.51).
- Layer 2 routes all 50 nurse-days and all 203 visits, with at least 3.2 minutes to spare on every route.
- The trade-off is slightly less continuity: 12 patients see more than one nurse instead of 8.

The fix uses one return trip as a stand-in for the real route, so it is not guaranteed for every future dataset. Layer 2's unroutable list stays the check that it held.

**No time windows in the forecast data.** Every visit can happen any time during the shift, so the waiting-time analysis can't be shown on the forecast week. It runs automatically once windows are added to the appointments file.
