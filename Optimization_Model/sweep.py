"""
Staffing sweep: what continuity of care costs, at each roster size.

For each roster size and each continuity weight w_c:
  1. Layer 1 (model.py) assigns patients to nurses for the week
  2. Layer 2 (sequencing.py) routes every nurse-day
  3. record actual route cost from Layer 2 and continuity from the assignment

Which nurses are dropped matters as much as how many: nurses differ in home base and days off.
So each roster size is drawn several times (one seeded shuffle per draw; a roster of k nurses is
the first k of it), and the sweep reports how many draws are feasible and the average result.

Output
  Outputs/staffing_sweep.csv   one row per (roster size, draw, w_c)
Charts are drawn from it by Optimization_Model/figures.py.

Run from the repo root:  python -m Optimization_Model.sweep   then   python -m Optimization_Model.figures
"""

import time

import numpy as np
import pandas as pd
from gurobipy import GRB

from Optimization_Model.model import build_model
from Optimization_Model import sequencing as seq

DATA_DIR = './Datasets/Forecast/' # same inputs as main.py
TRAVEL = './Datasets/Processed/travel_car.csv'
OUTPUT_DIR = seq.OUTPUT_DIR

ROSTER_SIZES = [7, 8, 9, 10, 11]
W_C_VALUES = [0, 1, 2, 5, 10, 25, 50, 100, 500]
ROSTER_SEEDS = [0, 1, 2, 3, 4] # one roster draw per seed, at every size
TIME_LIMIT = 300 # seconds per assignment solve


def roster_order(nurse_ids, seed):
    """One seeded shuffle; a roster of k nurses is the first k."""
    return list(np.random.default_rng(seed).permutation(sorted(nurse_ids)))


def assignments_from_x(x):
    """{(nurse, day): [patient, ...]} from the solved assignment variables."""
    assigned = {}
    for (i, j, d), var in x.items():
        if var.X > 0.5:
            assigned.setdefault((i, d), []).append(j)
    return assigned


def continuity_metrics(assigned):
    """Continuity of care for the week, computed from x (not z: with w_c = 0 nothing stops the
    solver setting extra z's to 1).

    single_nurse_pct      % of patients who see the same nurse for every visit
    patients_multi_nurse  number of patients who see more than one nurse
    distinct_nurses       average number of different nurses per patient (1.00 = full continuity)
    primary_share         share of all visits made by each patient's main nurse
    """
    visits = pd.DataFrame([(nurse, patient) for (nurse, _), patients in assigned.items() for patient in patients],
                          columns=['nurse_id', 'patient_id'])
    nurses_per_patient = visits.groupby('patient_id')['nurse_id'].nunique()
    by_nurse = visits.groupby(['patient_id', 'nurse_id']).size()
    return {
        'single_nurse_pct': round(100 * (nurses_per_patient == 1).mean(), 1),
        'patients_multi_nurse': int((nurses_per_patient > 1).sum()),
        'distinct_nurses': round(nurses_per_patient.mean(), 3),
        'primary_share': round(by_nurse.groupby('patient_id').max().sum() / len(visits), 3),
    }


def run_one(roster, seed, w_c, nurse_avail, patient_appt, dist, visits, travel_min, miles, shifts):
    row = {'nurse_count': len(roster), 'seed': seed, 'w_c': w_c, 'roster': ' '.join(sorted(roster))}

    model, x, z = build_model(nurse_avail[nurse_avail['nurse_id'].isin(roster)], patient_appt, dist, w_c)
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = TIME_LIMIT
    start = time.time()
    model.optimize()
    row['solve_seconds'] = round(time.time() - start, 1)

    if model.SolCount == 0: # infeasible, or no solution within the time limit
        row['status'] = 'infeasible' if model.Status == GRB.INFEASIBLE else f'no solution (status {model.Status})'
        return row
    row['status'] = 'optimal' if model.Status == GRB.OPTIMAL else f'time limit (gap {model.MIPGap:.1%})'
    row['l1_objective'] = round(model.ObjVal, 2)

    assigned = assignments_from_x(x)
    schedule, summary, unroutable = seq.run_sequencing(assigned, visits, travel_min, miles, shifts)
    seq.check_coverage(assigned, visits, schedule, unroutable)

    row.update(continuity_metrics(assigned))
    row['total_miles'] = round(summary['route_miles'].sum(), 1) if len(summary) else 0.0
    row['total_cost'] = round(summary['route_cost'].sum(), 2) if len(summary) else 0.0
    row['wait_minutes'] = round(summary['route_wait_minutes'].sum(), 1) if len(summary) else 0.0
    row['unroutable_days'] = len(unroutable)
    row['heuristic_routes'] = int((summary['method'] == 'heuristic').sum()) if len(summary) else 0
    row['max_stops'] = int(summary['stops'].max()) if len(summary) else 0
    row['min_slack_min'] = round(summary['slack_min'].min(), 1) if len(summary) else None
    return row


def main():
    nurse_avail = pd.read_csv(DATA_DIR + 'nurse_availability.csv')
    patient_appt = pd.read_csv(DATA_DIR + 'patient_appointments.csv')
    dist = pd.read_csv(TRAVEL)

    visits = seq.load_appointments()
    travel_min, miles = seq.load_travel()
    shifts = seq.load_shifts()

    orders = {seed: roster_order(nurse_avail['nurse_id'].unique(), seed) for seed in ROSTER_SEEDS}

    results = []
    for count in ROSTER_SIZES:
        for seed, order in orders.items():
            roster = order[:count]
            for w_c in W_C_VALUES:
                row = run_one(roster, seed, w_c, nurse_avail, patient_appt, dist, visits, travel_min, miles, shifts)
                results.append(row)
                print(f"{count:>2} nurses  draw {seed}  w_c={w_c:<4} {row['status']:<12} "
                      + (f"${row['total_cost']:>7.2f}  {row['single_nurse_pct']:5.1f}% keep one nurse  "
                         f"unroutable {row['unroutable_days']}  ({row['solve_seconds']}s)" if 'total_cost' in row else ''))

    results = pd.DataFrame(results)
    feasible = (results.assign(ok=results['status'].str.startswith(('optimal', 'time limit')))
                .groupby(['nurse_count', 'seed'])['ok'].all().groupby('nurse_count').sum())
    print('\nRoster draws that can be assigned, by size:',
          ', '.join(f'{k} nurses {v}/{len(ROSTER_SEEDS)}' for k, v in feasible.items()))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_DIR / 'staffing_sweep.csv', index=False)
    print(f"Saved {OUTPUT_DIR / 'staffing_sweep.csv'}  (charts: python -m Optimization_Model.figures)")


if __name__ == '__main__':
    main()
