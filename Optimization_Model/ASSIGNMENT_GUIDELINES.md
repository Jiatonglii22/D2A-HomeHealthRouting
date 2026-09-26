# Getting the Assignment Model Working

Live version with comments: https://claude.ai/code/artifact/c686ece1-a617-49d4-a58b-fe945f8f34a7

## 1. Fix the three crashes

The current `Optimization_Model/model.py` stops before it reaches the solver. With these three changes, it solves the dummy week to optimal (we tested this on a patched copy).

**a. `l` is never initialized for each patient** — `KeyError: 'P004'` at line 46. Initialize it next to `P`:

```python
for patient in patients:
    P[patient] = {}
    l[patient] = {}   # add this
```

**b. Wrong Gurobi indexing in the travel-cost objective** (line 134). Gurobi `tupledict`s are indexed by a tuple, not nested:

```python
# before
c[i][j] * x[i][j][d]
# after
c[i][j] * x[i,j,d]
```

**c. Days off have no shift times** (lines 27–28). A row with `available = 0` has blank `start_time`/`end_time`, so `time_to_minutes` returns `None` and `end - start` crashes in the `hours_worked` constraint. Default to 0:

```python
start[nurse][day] = time_to_minutes(match.iloc[0]['start_time']) or 0
end[nurse][day] = time_to_minutes(match.iloc[0]['end_time']) or 0
```

## 2. Test data and what the model reads

Use `Datasets/Dummy/`: 6 nurses, 25 patients and 52 appointments over one Mon–Fri week in Pittsburgh. Regenerate it with `python Datasets/generate_dummy_data.py` (seeded). The full data dictionary is in `Datasets/Dummy/README.md`. The files also carry columns for sequencing, which assignment can ignore.

| File | Columns model.py reads | Rules the data follows |
| --- | --- | --- |
| `nurse_availability.csv` | nurse_id, day, available, start_time, end_time | Day labels are `Mon Tues Wed Thurs Fri`, matching `days` in model.py. Days off have blank times. Shifts are ≤ 8 h. |
| `patient_appointments.csv` | patient_id, day, service_time | At most one visit per patient per day. |
| `travel_matrix.csv` | origin, destination, distance_miles | Cost is looked up as `c[nurse_id][patient_id]`, so the nurse's home base is the origin, not `Agency`. Every ordered pair is included. |

Two naming traps:

- The schema example says `Tue`, but model.py says `Tues`. A mismatched label is silently dropped: that day's appointments get `P = 0`, so the model just skips them. Normalize day labels on load, or assert they are all in `days`.
- If the real data only has `Agency` → patient rows, `c[i][j]` throws `KeyError` for every nurse. Decide whether travel cost starts from the agency or from home, and key it consistently.

## 3. Wire up main.py

`main.py` calls `build_model()` with no arguments and treats the result as a model, but `build_model` takes four inputs and returns `(model, x, z)`. Replace it with:

```python
import pandas as pd
from gurobipy import GRB
from Optimization_Model.model import build_model, time_to_minutes

DATA = 'Datasets/Dummy/'
nurse_avail = pd.read_csv(DATA + 'nurse_availability.csv')
patient_appt = pd.read_csv(DATA + 'patient_appointments.csv')
dist = pd.read_csv(DATA + 'travel_matrix.csv')

model, x, z = build_model(nurse_avail, patient_appt, dist, w_c=5)
model.optimize()

if model.Status == GRB.INFEASIBLE:
    model.computeIIS()
    model.write('infeasible.ilp')   # lists the conflicting constraints
assert model.Status == GRB.OPTIMAL, model.Status
```

The model is 900 variables and 1,685 constraints, which is under the 2,000 limit of the free pip `gurobipy` license. Anything bigger (for example about 8+ nurses at 25 patients) needs a free academic license.

## 4. Modeling issues to fix next

These don't crash the model, but they give wrong or misleading answers. They're ordered by impact.

1. **No skill constraint.** On the dummy week, the model gave P016's Wednesday recertification (RN-only) to N004, an LPN. Block those pairs using `required_discipline` from the appointments and `discipline` from `nurses.csv`:

    ```python
    model.addConstrs((x[i,j,d] == 0 for (i,j,d) in rn_only_pairs_for_lpns), name='skill')
    ```

2. **`limit_shift_hours` is not a real constraint.** `end - start <= 480` has no decision variables, so it evaluates to a Python `True`/`False`. We tested it: Gurobi accepts a `False` silently and the whole model becomes INFEASIBLE, with no hint why. Replace it with an input check: `assert (end - start <= 480).all()`.
3. **Capacity ignores routing.** `hours_worked` charges a flat 30 minutes of travel per visit and knows nothing about time windows or visit caps. So an assignment can fit on hours and still be impossible to route, for example two 9:00–10:00 visits for one nurse. Cheap fixes that keep assignment linear:
    - a visit cap: `sum_j x[i,j,d] <= max_visits_per_day[i]`
    - AM-window load ≤ the nurse's shift minutes before noon (PM likewise)
    - at most one `FIXED` window per nurse per start hour
4. **Continuity only counts this week.** `z` keeps a patient with one nurse within the week, but ignores who saw them before. Only 2 of 21 returning patients kept their `prior_nurse_id`. Add a penalty for `z[i,j] = 1` where `i` is not the prior nurse, or give `z[prior, j]` a bonus.
5. **Travel cost is home → patient for every visit.** It rewards nearby patients but not visits that chain well together. That's fine for assignment; sequencing fixes it. Remember it when reading the sensitivity results: `w_c = 5` means one extra nurse per patient is worth $5, or 10 miles of driving at $0.50/mile.

## 5. Done when

Assignment works when these checks pass on the dummy week. Run them after `model.optimize()` in main.py (`time_to_minutes` is imported from model.py):

```python
assign = pd.DataFrame([(i, j, d) for (i, j, d), v in x.items() if v.X > 0.5],
                      columns=['nurse_id', 'patient_id', 'day'])
assign = assign.merge(patient_appt, on=['patient_id', 'day'])

assert model.Status == GRB.OPTIMAL
assert len(assign) == len(patient_appt)                                   # every visit assigned once
on = nurse_avail[nurse_avail['available'] == 1]
assert len(assign.merge(on, on=['nurse_id', 'day'])) == len(assign)       # only on working days
load = (assign.assign(m=assign['service_time'] + 30)
        .groupby(['nurse_id', 'day'])['m'].sum().reset_index()
        .merge(on, on=['nurse_id', 'day']))
shift = load['end_time'].map(time_to_minutes) - load['start_time'].map(time_to_minutes)
assert (load['m'] <= shift).all()                                          # fits in shift
nurses = pd.read_csv(DATA + 'nurses.csv')
bad = assign.merge(nurses, on='nurse_id').query("required_discipline == 'RN' and discipline == 'LPN'")
assert bad.empty, bad[['appointment_id', 'nurse_id']]                     # skills (after fix 4.1)
```

These are the results from our patched copy with `w_c = 5`. Use them to confirm you get the same answer:

| Version | Status | Objective | Patients with >1 nurse | Solve time |
| --- | --- | --- | --- | --- |
| Section 1 fixes only | OPTIMAL | 79.00 | 0 | 0.1 s |
| + skill constraint (4.1) | OPTIMAL | 79.75 | 0 | 0.1 s |

With only the section 1 fixes, every check passes except the skill check, which flags A027 (P016's Wednesday recert) assigned to N004.

## 6. Handoff to sequencing

Sequencing runs next, on the output of assignment: for each nurse and day, it puts the assigned visits in order and sets arrival times. The `assign` table from section 5 is the handoff. Keep these columns, which already come from `patient_appointments.csv`:

| Column | Why sequencing needs it |
| --- | --- |
| nurse_id, day | one route per nurse-day |
| appointment_id, patient_id | stops on the route |
| service_time | time spent at each stop |
| tw_start, tw_end | the visit must start inside this window |
| priority | which visit to drop first if a route can't fit everything |

Sequencing also reads `nurses.csv` (the home base, where each route starts and ends) and `travel_matrix.csv` (`travel_minutes` for every ordered pair, including patient → patient and patient → home).

If a nurse-day turns out impossible to route, feed that back to assignment. For example, forbid that exact set of visits for that nurse and re-solve, rather than patching routes by hand. The section 4.3 capacity constraints make that loop rare.
