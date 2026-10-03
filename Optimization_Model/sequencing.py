"""
Layer 2 - Sequencing.

Takes Layer 1's nurse-day assignments (Outputs/nurse_daily_assignments.csv) and finds the
visit order and timetable for every nurse-day:
  - up to MAX_EXACT_STOPS stops: try every ordering (exact)
  - above that: nearest neighbour, then 2-opt improvement (heuristic)

Feasibility, checked for every ordering:
  - arriving before a visit's window opens means waiting, not starting early
  - a visit must start no later than its window closes
  - the nurse must be back home by the end of her shift

Inputs
  Outputs/nurse_daily_assignments.csv                    Layer 1 output (which nurse sees whom, each day)
  Datasets/Processed/synthetic_appointments_397636.csv   visit length per (patient, day)
  Datasets/Processed/travel_car.csv                      travel minutes / miles between every pair of locations
  Datasets/Forecast/nurse_availability.csv               shifts (the same shifts Layer 1 solved with;
                                                         no shift data exists in Processed)

Outputs (in Outputs/)
  route_schedule.csv            one row per stop
  route_schedule_<day>.csv      the same, one file per day
  route_summary.csv             one row per nurse-day
  unroutable_nurse_days.csv     nurse-days with no feasible ordering, and why

Run from the repo root, after main.py:  python -m Optimization_Model.sequencing
"""

import math
from itertools import permutations
from pathlib import Path

import pandas as pd

ASSIGNMENTS = Path('./Outputs/nurse_daily_assignments.csv')
APPOINTMENTS = Path('./Datasets/Processed/synthetic_appointments_397636.csv')
TRAVEL = Path('./Datasets/Processed/travel_car.csv')
SHIFTS = Path('./Datasets/Forecast/nurse_availability.csv')
OUTPUT_DIR = Path('./Outputs')

DAYS = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri'] # same labels as model.py
DAY_MAP = {'Monday': 'Mon', 'Tuesday': 'Tues', 'Wednesday': 'Wed', 'Thursday': 'Thurs', 'Friday': 'Fri'}

COST_PER_MILE = 0.235 # keep in sync with cpg in model.py
MAX_EXACT_STOPS = 8 # 8! = 40,320 orderings; 9! = 362,880


# ---------------------------------------------------------------- inputs

def time_to_minutes(time_value):
    if pd.isna(time_value):
        return None
    hours, minutes = map(int, str(time_value).split(':'))
    return (hours * 60) + minutes


def minutes_to_time(minutes):
    minutes = int(round(minutes))
    return f'{minutes // 60}:{minutes % 60:02d}'


def load_assignments(path=ASSIGNMENTS):
    """{(nurse, day): [patient, ...]} from Layer 1's wide CSV (one column per day)."""
    wide = pd.read_csv(path)
    assigned = {}
    for _, row in wide.iterrows():
        for day in DAYS:
            cell = row.get(day)
            if pd.isna(cell) or not str(cell).strip():
                continue
            assigned[(row['Nurse_ID'], day)] = [p.strip() for p in str(cell).split(',') if p.strip()]
    return assigned


def load_appointments(path=APPOINTMENTS):
    """Per (patient, day): appointment id, visit length, and arrival window.

    Keyed on (patient, day), not patient: a patient can need 30 minutes on Monday and 60 on Wednesday.
    The forecast has no time windows; if tw_start / tw_end columns are added later they are used,
    otherwise the window is left open (None) and only the shift limits the visit.
    """
    appts = pd.read_csv(path)
    appts['day'] = appts['day'].map(DAY_MAP).fillna(appts['day'])
    has_windows = {'tw_start', 'tw_end'} <= set(appts.columns)

    visits = {}
    for row in appts.itertuples(index=False):
        visits[(row.patient_id, row.day)] = {
            'appointment_id': row.appointment_id,
            'service': float(row.service_time_minutes),
            'tw_start': time_to_minutes(row.tw_start) if has_windows else None,
            'tw_end': time_to_minutes(row.tw_end) if has_windows else None,
        }
    return visits


def load_travel(path=TRAVEL):
    """travel_min[a][b], miles[a][b] for every ordered pair (Agency, nurse homes, patients)."""
    travel = pd.read_csv(path)
    travel_min, miles = {}, {}
    for row in travel.itertuples(index=False):
        travel_min.setdefault(row.origin, {})[row.destination] = float(row.travel_minutes)
        miles.setdefault(row.origin, {})[row.destination] = float(row.distance_miles)
    return travel_min, miles


def load_shifts(path=SHIFTS):
    """{(nurse, day): (start_minute, end_minute)} for working days only."""
    avail = pd.read_csv(path)
    return {
        (row.nurse_id, row.day): (time_to_minutes(row.start_time), time_to_minutes(row.end_time))
        for row in avail.itertuples(index=False)
        if row.available == 1
    }


# ---------------------------------------------------------------- routing

def evaluate(order, day, depot, shift_start, shift_end, visits, travel_min, miles, best_miles=math.inf, keep_stops=False):
    """Drive one ordering and report what happens.

    Returns a dict with status ('ok', 'window', 'overrun', 'pruned'), miles, drive/wait minutes,
    return time, and (if keep_stops) one record per stop. Stops early once the window is missed
    or the running miles reach best_miles (that ordering can't win).
    """
    t, loc = shift_start, depot
    total_miles = drive = wait = 0.0
    stops = []

    for p in order:
        visit = visits[(p, day)]
        leg_min, leg_miles = travel_min[loc][p], miles[loc][p]
        t += leg_min # drive there
        drive += leg_min
        total_miles += leg_miles
        if total_miles >= best_miles:
            return {'status': 'pruned'}

        arrive = t
        if visit['tw_start'] is not None and t < visit['tw_start']: # wait if early
            t = visit['tw_start']
        if visit['tw_end'] is not None and t > visit['tw_end']: # window missed
            return {'status': 'window', 'patient': p, 'late_by': t - visit['tw_end']}
        wait += t - arrive

        if keep_stops:
            stops.append({'patient_id': p, 'appointment_id': visit['appointment_id'], 'arrive': arrive, 'start': t,
                          'wait_min': t - arrive, 'end': t + visit['service'],
                          'drive_min_from_prev': leg_min, 'miles_from_prev': leg_miles})
        t += visit['service'] # do the visit
        loc = p

    t += travel_min[loc][depot] # drive home
    drive += travel_min[loc][depot]
    total_miles += miles[loc][depot]

    result = {'miles': total_miles, 'drive_min': drive, 'wait_min': wait, 'return_time': t, 'stops': stops,
              'overrun_by': t - shift_end}
    result['status'] = 'ok' if t <= shift_end else 'overrun' # shift overrun
    return result


def best_route(patients, day, depot, shift_start, shift_end, visits, travel_min, miles):
    """Exact: try every ordering and keep the cheapest feasible one (by miles).

    If none is feasible, return the reason from the closest attempt (smallest overrun, else a missed window).
    """
    best_order, best_miles = None, math.inf
    closest_overrun, missed_window = None, None

    for order in permutations(patients):
        r = evaluate(order, day, depot, shift_start, shift_end, visits, travel_min, miles, best_miles)
        if r['status'] == 'ok' and r['miles'] < best_miles: # pruning skips the drive home, so compare full miles
            best_order, best_miles = order, r['miles']
        elif r['status'] == 'overrun' and (closest_overrun is None or r['overrun_by'] < closest_overrun):
            closest_overrun = r['overrun_by']
        elif r['status'] == 'window' and missed_window is None:
            missed_window = r

    if best_order is not None:
        return best_order, None
    if closest_overrun is not None:
        return None, f'shift overrun: best ordering returns {closest_overrun:.0f} min after shift end'
    return None, f"window missed: no ordering reaches every visit in time (e.g. {missed_window['patient']})"


def heuristic_route(patients, day, depot, shift_start, shift_end, visits, travel_min, miles):
    """Above MAX_EXACT_STOPS: nearest neighbour by miles, then 2-opt.

    Orderings are compared feasible-first, then by size of violation, then by miles, using the same
    evaluate() as the exact search so the two can never disagree about feasibility.
    """
    def score(order):
        r = evaluate(order, day, depot, shift_start, shift_end, visits, travel_min, miles)
        if r['status'] == 'ok':
            return (0, 0.0, r['miles'])
        if r['status'] == 'overrun':
            return (1, r['overrun_by'], r['miles'])
        return (2, r['late_by'], 0.0)

    # nearest neighbour
    order, remaining, loc = [], list(patients), depot
    while remaining:
        nxt = min(remaining, key=lambda p: miles[loc][p])
        order.append(nxt)
        remaining.remove(nxt)
        loc = nxt

    # 2-opt: reverse any segment that improves the score, until nothing does
    best, best_score = order, score(order)
    improved = True
    while improved:
        improved = False
        for a in range(len(best) - 1):
            for b in range(a + 1, len(best)):
                candidate = best[:a] + best[a:b + 1][::-1] + best[b + 1:]
                candidate_score = score(candidate)
                if candidate_score < best_score:
                    best, best_score = candidate, candidate_score
                    improved = True

    if best_score[0] == 0:
        return tuple(best), None
    if best_score[0] == 1:
        return None, f'shift overrun: best ordering found returns {best_score[1]:.0f} min after shift end (heuristic)'
    return None, 'window missed: no ordering found reaches every visit in time (heuristic)'


def run_sequencing(assigned, visits, travel_min, miles, shifts, depot=None):
    """Route every nurse-day.

    depot=None starts and ends each route at the nurse's home (the cost basis Layer 1 uses);
    depot='Agency' starts and ends every route at the agency instead, for comparison.

    Returns (schedule, summary, unroutable) DataFrames.
    """
    schedule_rows, summary_rows, unroutable_rows = [], [], []

    for (nurse, day), patients in sorted(assigned.items(), key=lambda k: (DAYS.index(k[0][1]), k[0][0])):
        start, end = shifts[(nurse, day)]
        origin = depot or nurse
        exact = len(patients) <= MAX_EXACT_STOPS
        router = best_route if exact else heuristic_route
        order, reason = router(patients, day, origin, start, end, visits, travel_min, miles)

        if order is None:
            unroutable_rows.append({'nurse_id': nurse, 'day': day, 'stops': len(patients),
                                    'patients': ', '.join(patients), 'reason': reason})
            continue

        r = evaluate(order, day, origin, start, end, visits, travel_min, miles, keep_stops=True)
        for n, stop in enumerate(r['stops'], start=1):
            schedule_rows.append({
                'nurse_id': nurse, 'day': day, 'stop': n,
                'patient_id': stop['patient_id'], 'appointment_id': stop['appointment_id'],
                'arrive': minutes_to_time(stop['arrive']), 'start': minutes_to_time(stop['start']),
                'end': minutes_to_time(stop['end']), 'wait_min': round(stop['wait_min'], 1),
                'drive_min_from_prev': round(stop['drive_min_from_prev'], 1),
                'miles_from_prev': round(stop['miles_from_prev'], 2),
            })
        summary_rows.append({
            'nurse_id': nurse, 'day': day, 'stops': len(order), 'method': 'exact' if exact else 'heuristic',
            'shift_start': minutes_to_time(start), 'shift_end': minutes_to_time(end),
            'return_time': minutes_to_time(r['return_time']), 'slack_min': round(end - r['return_time'], 1),
            'route_miles': round(r['miles'], 2), 'route_cost': round(r['miles'] * COST_PER_MILE, 2),
            'route_drive_minutes': round(r['drive_min'], 1), 'route_wait_minutes': round(r['wait_min'], 1),
            'route': ' > '.join([origin, *order, origin]),
        })

    schedule = pd.DataFrame(schedule_rows)
    summary = pd.DataFrame(summary_rows)
    unroutable = pd.DataFrame(unroutable_rows, columns=['nurse_id', 'day', 'stops', 'patients', 'reason'])
    return schedule, summary, unroutable


def check_coverage(assigned, visits, schedule, unroutable):
    """Every appointment is assigned exactly once, and every assigned visit is routed or reported."""
    assigned_visits = [(p, day) for (_, day), patients in assigned.items() for p in patients]
    missing = set(visits) - set(assigned_visits)
    unknown = set(assigned_visits) - set(visits)
    assert not unknown, f'assigned visits with no appointment: {sorted(unknown)[:5]}'
    assert not missing, f'appointments Layer 1 did not assign: {sorted(missing)[:5]}'
    assert len(assigned_visits) == len(set(assigned_visits)), 'a visit is assigned to more than one nurse'

    routed = len(schedule)
    reported = unroutable['stops'].sum() if len(unroutable) else 0
    assert routed + reported == len(assigned_visits), f'{routed} routed + {reported} unroutable != {len(assigned_visits)} assigned'


# ---------------------------------------------------------------- main

def main():
    assigned = load_assignments()
    visits = load_appointments()
    travel_min, miles = load_travel()
    shifts = load_shifts()

    schedule, summary, unroutable = run_sequencing(assigned, visits, travel_min, miles, shifts)
    check_coverage(assigned, visits, schedule, unroutable)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(OUTPUT_DIR / 'route_schedule.csv', index=False)
    for day in DAYS:
        schedule[schedule['day'] == day].to_csv(OUTPUT_DIR / f'route_schedule_{day}.csv', index=False)
    summary.to_csv(OUTPUT_DIR / 'route_summary.csv', index=False)
    unroutable.to_csv(OUTPUT_DIR / 'unroutable_nurse_days.csv', index=False)

    print(f"Routed {len(summary)} nurse-days ({(summary['method'] == 'heuristic').sum()} heuristic), "
          f"{len(schedule)} visits")
    print(f"Total: {summary['route_miles'].sum():.1f} miles, ${summary['route_cost'].sum():.2f}, "
          f"{summary['route_drive_minutes'].sum():.0f} drive min, {summary['route_wait_minutes'].sum():.0f} wait min")
    if len(unroutable):
        print(f'\n{len(unroutable)} unroutable nurse-days (Layer 1 gave a caseload that cannot be driven inside the shift):')
        print(unroutable[['nurse_id', 'day', 'stops', 'reason']].to_string(index=False))


if __name__ == '__main__':
    main()
