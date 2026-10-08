import math
from itertools import permutations
from pathlib import Path
import pandas as pd

# input and output path definitions 
DATA_DIR = './Datasets/Forecast/'
OUTPUT_DIR = './Outputs/'

ASSIGNMENTS = Path(OUTPUT_DIR + 'nurse_daily_assignments.csv')
APPOINTMENTS = Path(DATA_DIR + 'patient_appointments.csv')
SHIFTS = Path(DATA_DIR + 'nurse_availability.csv')
TRAVEL = Path('./Datasets/Processed/travel_car.csv')

DAYS = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri'] 
DAY_MAP = {'Monday': 'Mon', 'Tuesday': 'Tues', 'Wednesday': 'Wed', 'Thursday': 'Thurs', 'Friday': 'Fri'}

COST_PER_MILE = 0.235 
MAX_EXACT_STOPS = 8 

# converts a given time to minutes 
def time_to_minutes(time_value):
    if pd.isna(time_value):
        return None
    
    hours, minutes = map(int, str(time_value).split(':'))
    return (hours * 60) + minutes

# converts a given number of minutes to a time 
def minutes_to_time(minutes):
    minutes = int(round(minutes))
    return f'{minutes // 60}:{minutes % 60:02d}'

# load the assignment output from the assignment optimization model 
def load_assignments(path=ASSIGNMENTS):
    wide = pd.read_csv(path)
    assigned = {}

    for _, row in wide.iterrows():
        for day in DAYS:
            cell = row.get(day)
            if pd.isna(cell) or not str(cell).strip():
                continue
            assigned[(row['Nurse_ID'], day)] = [p.strip() for p in str(cell).split(',') if p.strip()]
    return assigned

# load the patient appointment data 
def load_appointments(path=APPOINTMENTS):
    appts = pd.read_csv(path)
    appts['day'] = appts['day'].map(DAY_MAP).fillna(appts['day'])

    visits = {}
    for row in appts.itertuples(index=False):
        visits[(row.patient_id, row.day)] = {
            'appointment_id': row.appointment_id,
            'service': float(row.service_time)
        }

    return visits

# load the distance matrix 
def load_travel(path=TRAVEL):
    travel = pd.read_csv(path)
    travel_min, miles = {}, {}

    for row in travel.itertuples(index=False):
        travel_min.setdefault(row.origin, {})[row.destination] = float(row.travel_minutes)
        miles.setdefault(row.origin, {})[row.destination] = float(row.distance_miles)

    return travel_min, miles

# load nurse availability and shift times 
def load_shifts(path=SHIFTS):
    avail = pd.read_csv(path)
    shifts = {} 

    for row in avail.itertuples(index=False):
        shifts[(row.nurse_id, row.day)] = {
            'start_time': row.start_time, 
            'end_time': row.end_time
        }

    return shifts


# try one ordering of routes and report whether it was successful or improved on from the previous ordering in terms of driving distance and time 
def evaluate(order, day, depot, shift_start, shift_end, visits, travel_min, miles, best_miles=math.inf, keep_stops=False):
    t, loc = shift_start, depot
    total_miles = drive = 0.0
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

        # record stop information 
        if keep_stops:
            stops.append({'patient_id': p, 'appointment_id': visit['appointment_id'], 'arrive': arrive, 'start': t,
                          'end': t + visit['service'], 'drive_min_from_prev': leg_min, 'miles_from_prev': leg_miles})
        t += visit['service'] # do the visit
        loc = p # update location to current stop 

    t += travel_min[loc][depot] # drive home
    # keep track of total travel time and miles 
    drive += travel_min[loc][depot]
    total_miles += miles[loc][depot]

    result = {'miles': total_miles, 'drive_min': drive, 'return_time': t, 'stops': stops,
              'overrun_by': t - shift_end}
    result['status'] = 'ok' if t <= shift_end else 'overrun' # shift overrun
    return result

# try every ordering and keep the cheapest and most feasible route (by miles) 
# if no ordering is feasible, return the one with the smallest overrun shift time 
def best_route(patients, day, depot, shift_start, shift_end, visits, travel_min, miles):
    best_order, best_miles = None, math.inf
    closest_overrun = None

    # try each order and keep the best one 
    for order in permutations(patients):
        r = evaluate(order, day, depot, shift_start, shift_end, visits, travel_min, miles, best_miles)
        if r['status'] == 'ok' and r['miles'] < best_miles: 
            best_order, best_miles = order, r['miles']
        elif r['status'] == 'overrun' and (closest_overrun is None or r['overrun_by'] < closest_overrun):
            closest_overrun = r['overrun_by']

    # return ordering 
    if best_order is not None:
        return best_order, None
    if closest_overrun is not None:
        return None, f'shift overrun: best ordering returns {closest_overrun:.0f} min after shift end'

# if the number of patients will go over a nurse's shift, find the best one and choose the one with the least violation 
def heuristic_route(patients, day, depot, shift_start, shift_end, visits, travel_min, miles):
    
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

# route each nurse shift 
# depot = None default value assumes nurse begins and ends the route at their home 
#       = 'Agency' assumes nurse begins and ends routes from the Agency 
def run_sequencing(assigned, visits, travel_min, miles, shifts, depot=None):
    schedule_rows, summary_rows, unroutable_rows = [], [], []

    # for each day, for each nurse, find the most optimal route 
    for (nurse, day), patients in sorted(assigned.items(), key=lambda k: (DAYS.index(k[0][1]), k[0][0])):
        start = time_to_minutes(shifts[(nurse, day)]['start_time'])
        end = time_to_minutes(shifts[(nurse, day)]['end_time'])
        origin = depot or nurse
        exact = len(patients) <= MAX_EXACT_STOPS
        router = best_route if exact else heuristic_route
        order, reason = router(patients, day, origin, start, end, visits, travel_min, miles)

        if order is None:
            unroutable_rows.append({'nurse_id': nurse, 'day': day, 'stops': len(patients),
                                    'patients': ', '.join(patients), 'reason': reason})
            continue

        r = evaluate(order, day, origin, start, end, visits, travel_min, miles, keep_stops=True)
        # record data about best route 
        for n, stop in enumerate(r['stops'], start=1):
            schedule_rows.append({
                'nurse_id': nurse, 'day': day, 'stop': n,
                'patient_id': stop['patient_id'], 'appointment_id': stop['appointment_id'],
                'arrive': minutes_to_time(stop['arrive']), 'start': minutes_to_time(stop['start']),
                'end': minutes_to_time(stop['end']),
                'drive_min_from_prev': round(stop['drive_min_from_prev'], 1),
                'miles_from_prev': round(stop['miles_from_prev'], 2),
            })
        summary_rows.append({
            'nurse_id': nurse, 'day': day, 'stops': len(order), 'method': 'exact' if exact else 'heuristic',
            'shift_start': minutes_to_time(start), 'shift_end': minutes_to_time(end),
            'return_time': minutes_to_time(r['return_time']), 'slack_min': round(end - r['return_time'], 1),
            'route_miles': round(r['miles'], 2), 'route_cost': round(r['miles'] * COST_PER_MILE, 2),
            'route_drive_minutes': round(r['drive_min'], 1), 
            'route': ' > '.join([origin, *order, origin]),
        })

    schedule = pd.DataFrame(schedule_rows)
    summary = pd.DataFrame(summary_rows)
    unroutable = pd.DataFrame(unroutable_rows, columns=['nurse_id', 'day', 'stops', 'patients', 'reason'])
    return schedule, summary, unroutable

# ensure that every patient appointment has been assigned once (covered by assignment model) 
# every assigned visit should be included in a route 
def check_coverage(assigned, visits, schedule, unroutable):
    assigned_visits = [(p, day) for (_, day), patients in assigned.items() for p in patients]
    missing = set(visits) - set(assigned_visits)
    unknown = set(assigned_visits) - set(visits)
    assert not unknown, f'assigned visits with no appointment: {sorted(unknown)[:5]}'
    assert not missing, f'appointments Layer 1 did not assign: {sorted(missing)[:5]}'
    assert len(assigned_visits) == len(set(assigned_visits)), 'a visit is assigned to more than one nurse'

    routed = len(schedule)
    reported = unroutable['stops'].sum() if len(unroutable) else 0
    assert routed + reported == len(assigned_visits), f'{routed} routed + {reported} unroutable != {len(assigned_visits)} assigned'


def sequence_main():
    # load inputs 
    assigned = load_assignments()
    visits = load_appointments()
    travel_min, miles = load_travel()
    shifts = load_shifts()

    # generate schedule + verify output 
    schedule, summary, unroutable = run_sequencing(assigned, visits, travel_min, miles, shifts)
    check_coverage(assigned, visits, schedule, unroutable)

    # output to files 
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    schedule.to_csv(OUTPUT_DIR + 'route_schedule.csv', index=False)
    for day in DAYS:
        schedule[schedule['day'] == day].to_csv(OUTPUT_DIR + f'route_schedule_{day}.csv', index=False)
    summary.to_csv(OUTPUT_DIR + 'route_summary.csv', index=False)
    unroutable.to_csv(OUTPUT_DIR + 'unroutable_nurse_days.csv', index=False)

    print(f"Routed {len(summary)} nurse-days ({(summary['method'] == 'heuristic').sum()} heuristic), "
          f"{len(schedule)} visits")
    print(f"Total: {summary['route_miles'].sum():.1f} miles, ${summary['route_cost'].sum():.2f}, "
          f"{summary['route_drive_minutes'].sum():.0f} drive min")
    if len(unroutable):
        print(f'\n{len(unroutable)} unroutable nurse-days:')
        print(unroutable[['nurse_id', 'day', 'stops', 'reason']].to_string(index=False))