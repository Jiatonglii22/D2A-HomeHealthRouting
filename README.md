# D2A-HomeHealthRouting
### Team Members: 
- Imama Zahoor 
- Jiatong Li 
- Esha Pandya 
- Vidhi Kothari 

*Decision question: For a home health agency serving Allegheny County, PA, how should weekly patient visits be assigned and sequenced across nurses, how many nurses of each skill type are required, and what is the operational cost of guaranteeing continuity of care?* 

## File Structure 
- Datasets: raw, forecasted, and processed datasets 
    - Processed datasets are ready to use as inputs 
- Optimization_Model: assignment and sequencing model 
    - Conducts sensitivity analysis and outputs results 
- Outputs: contains all outputs from optimization_model 
    - patient-nurse assignments per day 
    - daily and weekly schedules for each nurse 
    - route figures 
- Prediction_Model: uses existing data to perform demand allocation of patient appointments and locations 

## Setup: 
- Requires a Python version of 3.10 or higher 
- Download and unzip repository folder 
- Open a terminal in the main folder 
- Set up a virtual environment 
    - `python -m venv .venv` 
- Start virtual environment 
    - `source .venv/bin/activate` 
- Install required libraries 
    - `pip install -r requirements.txt` 

## Running the Models: 
- Run `main.py` 
    - `python main.py` 

## Outputs: 
- Navigate to `Outputs/` folder 
    - `nurse_daily_assignments.csv`: This csv contains the outputs of the assignment model. With columns as the day of the week and the row representing each nurse, each cell lists the IDs of the patients assigned to that nurse. It is the user readable output of the x_{ijd} binary decision variable. 
    - `route_schedule.csv`: This csv contains the output of the sequencing layer. Each row contains one stop in a nurse's stop, including the patient ID, patient appointment ID, day of the week, order in the stops, start and stop time of appointment, and drive time and mileage from the previous stop. 
    - `route_schedule_<day>.csv`: One csv per day is created in a similar format as the weekly schedule. 
    - `route_summary.csv`: This csv contains one row per nurse per day, with a list of patient IDs to easily extract the route. 
    - `unroutable_nurse_days.csv`: This csv will be populated if the sequencing model is unable to find a feasible route restricted to the nurse's shift hours. In an ideal situation, this file will be unpopulated. 
    - `staffing_sweep.csv`: This file contains the results of our sensitivty analysis, containing one row per combination of staff size and continuity of care weight. 

## Prediction Model

Agency-level demand allocation model is used to estimate agency-specific demand spatially using ACS need indicators, then use that estimated demand to create synthetic patients that Gurobi will route.

We're using UPMC Home Health Agency in Greater Pittsburgh Region as the agency we are allocating demand for in its serviced zip codes.

## Pipeline

The optimization runs in two layers:

1. **Assignment (Layer 1)**, `Optimization_Model/model.py`, run by `main.py`: a Gurobi MILP that decides which nurse sees which patient each day. It minimizes mileage cost from each nurse's home plus a continuity penalty `w_c` for every extra nurse a patient sees. Each nurse-day's visits, the drive to each one, and the drive home must fit within the shift.
2. **Sequencing (Layer 2)**, `Optimization_Model/sequencing.py`: takes those assignments and finds the visit order and timetable for every nurse-day.

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
| Nurse-days routed | 50 of 50, all by trying every order |
| Visits scheduled | 203 of 203 |
| Route miles / cost | 1,729.1 mi / $406.39 at $0.235 per mile |
| Unroutable nurse-days | none |
| Tightest route | N010 Wed, home 3.2 min before shift end |
| Patients seen by more than one nurse | 12 |
| Run time | under 1 second |

The code was checked against a brute-force search over every visit order: on all 50 nurse-days, the chosen route is the cheapest feasible one.

## Staffing sweep

`Optimization_Model/sweep.py` runs the whole pipeline many times, varying two things:

- **The continuity weight `w_c`:** 0, 1, 2, 5, 10, 25, 50, 100, 500.
- **The roster:** 7, 8, 9, 10 and 11 nurses. Each size is drawn 5 times at random (seeded), because which nurses are dropped matters as much as how many. Nurses differ in where they live and which days they work.

That is 5 sizes × 5 draws × 9 weights = 225 runs. Each run:

1. Solves assignment (Layer 1) with that roster and weight.
2. Routes every nurse-day (Layer 2).
3. Records the real result: route miles and cost from Layer 2, continuity from the assignment, and whether every nurse-day could be routed.

### Why we need it

**The continuity weight is a management judgement, not something the data settles.** No single value of `w_c` is right. It says how many dollars of driving one extra nurse per patient is worth, and that is the agency's call. A single run of the model gives one schedule for one arbitrary weight. The sweep instead shows the whole trade-off: what each step up in continuity costs in real driving. The agency can then pick its point on the curve.

**It answers the staffing question.** Running the same trade-off at each roster size shows:

- the smallest roster that can cover the week at all;
- whether that roster can still give every patient a single nurse;
- how much continuity costs as staff gets tighter.


### What it records (`staffing_sweep.csv`, one row per run)

| Column | Meaning |
|---|---|
| `nurse_count`, `seed`, `roster`, `w_c` | which run |
| `status` | `optimal`, or `infeasible` when the roster can't cover the week's visits within shifts |
| `total_miles`, `total_cost` | real weekly route miles and cost, from Layer 2 |
| `single_nurse_pct` | % of patients who see the same nurse for every visit that week (100% = full continuity) |
| `patients_multi_nurse` | number of patients who see more than one nurse |
| `distinct_nurses` | average number of different nurses per patient. Each patient sees a whole number of nurses (1, 2, …); this is the average over all patients, so 1.17 means roughly 1 patient in 6 sees two nurses. |
| `primary_share` | share of all visits made by each patient's main nurse |
| `unroutable_days` | nurse-days Layer 2 couldn't fit in the shift (should be 0) |
| `wait_minutes`, `heuristic_routes`, `max_stops`, `min_slack_min` | route checks: waiting for windows, routes not solved exactly, busiest nurse-day, tightest return home |
| `l1_objective`, `solve_seconds` | assignment's own objective and solve time |

Continuity is computed from the actual assignments (`x`), not from the model's continuity variables (`z`). With `w_c = 0` nothing stops the solver from switching on extra `z` values for free.

### Results

**Staffing: the week needs 9 nurses, or 8 if all eight work every day.**

| Roster | Can the week be covered? |
|---|---|
| 7 nurses | No, in all 5 random rosters. |
| 8 nurses | Not in any of the 5 random rosters, each of which included both nurses with a day off. In a separate check, 8 of the 9 eight-nurse rosters where everyone works all five days could cover it. |
| 9 nurses | Yes, all 5 rosters. In 2 of them one nurse-day couldn't be driven at low weights (see Limitations), so 3 of 5 rosters give a drivable schedule at every weight. |
| 10, 11 nurses | Yes, all 5 rosters, every nurse-day drivable. With 11 nurses every draw is the full roster, so the 5 draws are identical. |

**Continuity: most of it is nearly free, and the last few percent is what costs money.** Average over the rosters that give a drivable schedule at every weight:

| Roster | Lowest cost (`w_c = 0`) | Small weight (`w_c = 1`) | Full continuity |
|---|---|---|---|
| 9 nurses | $398, 81% keep one nurse | $407, 93% | $445, 100% (+$47, +12%) |
| 10 nurses | $395, 84% | $405, 94% | $426, 100% (+$31, +8%) |
| 11 nurses | $397, 85% | $403, 95% | $432, 100% (+$35, +9%) |

- **The first step is cheap.** Going from about 1 in 6 patients seeing two nurses to about 1 in 16 costs **$6–10 a week (about 2%)**.
- **The last step is expensive.** Getting the final 5–7% of patients down to a single nurse costs another **$21–38 a week**.
- **Tighter staffing makes full continuity dearer:** +12% at 9 nurses, +8–9% at 10–11.
- **Above `w_c = 25`, nothing changes:** every patient already keeps one nurse.


## Figures

`python -m Optimization_Model.figures` draws every chart from the saved CSVs into `Outputs/figures/`. 

| Figure | Built from | Shows |
|---|---|---|
| `staffing_sweep` | staffing sweep | what continuity costs, at each roster size |
| `cost_by_roster` | staffing sweep | lowest-cost vs full-continuity cost, per roster size |
| `nurse_time_breakdown` | current run (`w_c = 0.5`, 11 nurses) | how each nurse's week splits into visits, driving and spare time |
| `route_map_<day>` | current run | every nurse's route that day |

