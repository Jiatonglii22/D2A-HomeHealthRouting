"""
Generate a small, reproducible dummy dataset for the home health routing pipeline.

Covers both stages:
  - assignment (Optimization_Model/model.py) -- uses the core schema columns
  - sequencing / routing (not built yet)     -- uses the extra columns (time windows, skills, full travel matrix)

Outputs (in Datasets/Dummy/), see Datasets/Dummy/README.md for the data dictionary:
    nurse_availability.csv    nurse_id, day, available, start_time, end_time
    patient_appointments.csv  appointment_id, patient_id, day, location, lat, lon, service_time,
                              visit_type, required_discipline, tw_start, tw_end, priority
    travel_matrix.csv         origin, destination, travel_minutes, distance_miles   (every ordered pair of locations)
    nurses.csv                nurse_id, discipline, home_zip, lat, lon, max_visits_per_day, cost_per_mile
    patients.csv              patient_id, location, lat, lon, visits_per_week, window_type, tw_start, tw_end, prior_nurse_id

Setting: Pittsburgh, PA. 6 nurses, 25 patients, one Mon-Fri week.
Sized to stay under Gurobi's size-limited license (2000 vars / 2000 constraints) for the assignment model.

Run from repo root:  python Datasets/generate_dummy_data.py
"""

import math
import os

import numpy as np
import pandas as pd

SEED = 42
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Dummy')

# must match the day labels in model.py
DAYS = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri']

# approximate zip centroids around Pittsburgh (lat, lon)
ZIPS = {
    '15201': (40.4760, -79.9534),  # Lawrenceville
    '15203': (40.4274, -79.9796),  # South Side
    '15206': (40.4691, -79.9187),  # East Liberty
    '15207': (40.4032, -79.9345),  # Hazelwood / Greenfield
    '15208': (40.4550, -79.8990),  # Point Breeze
    '15210': (40.4060, -79.9870),  # Carrick
    '15211': (40.4295, -80.0143),  # Mt Washington
    '15212': (40.4596, -80.0101),  # North Side
    '15213': (40.4435, -79.9560),  # Oakland
    '15214': (40.4840, -80.0140),  # Perry
    '15215': (40.4990, -79.9150),  # Sharpsburg / O'Hara
    '15216': (40.4035, -80.0337),  # Dormont / Brookline
    '15217': (40.4318, -79.9227),  # Squirrel Hill
    '15218': (40.4235, -79.8889),  # Swissvale / Regent Square
    '15220': (40.4185, -80.0510),  # Green Tree
    '15221': (40.4349, -79.8680),  # Wilkinsburg
    '15222': (40.4476, -79.9933),  # Downtown
    '15224': (40.4648, -79.9448),  # Bloomfield
    '15232': (40.4521, -79.9316),  # Shadyside
    '15235': (40.4600, -79.8250),  # Penn Hills
}

AGENCY_ZIP = '15213'

# nurse_id, discipline, home zip, max visits/day, {day: (start, end)} -- days not listed are unavailable
NURSES = [
    ('N001', 'RN',  '15217', 6, {d: ('8:00', '16:00') for d in DAYS}),                             # full time
    ('N002', 'RN',  '15212', 6, {d: ('9:00', '17:00') for d in DAYS}),                             # full time, late start
    ('N003', 'RN',  '15203', 6, {d: ('7:00', '15:00') for d in ['Mon', 'Tues', 'Wed', 'Thurs']}),  # 4 days, early
    ('N004', 'LPN', '15206', 3, {d: ('8:00', '12:00') for d in ['Mon', 'Wed', 'Fri']}),            # part time mornings
    ('N005', 'LPN', '15216', 5, {d: ('10:00', '18:00') for d in ['Tues', 'Thurs']}),               # part time 2 days
    ('N006', 'RN',  '15221', 6, {d: ('8:30', '16:30') for d in ['Mon', 'Tues', 'Thurs', 'Fri']}),  # Wed off
]
COST_PER_MILE = 0.5  # matches cpg in model.py

N_PATIENTS = 25

# typical home-health visit frequencies -> day patterns
VISIT_PATTERNS = [
    (['Mon', 'Wed', 'Fri'], 0.25),   # 3x/week
    (['Tues', 'Thurs'], 0.20),       # 2x/week
    (['Mon', 'Thurs'], 0.15),        # 2x/week
    (['Wed', 'Fri'], 0.10),          # 2x/week
    (None, 0.30),                    # 1x/week, random day
]

# visit type -> (service minutes, required discipline, priority)
#   SOC = start of care (new patient, first visit), RECERT = 60-day recertification, DC = discharge
#   RN can cover any visit; LPN only ROUTINE. priority 1 = must happen that day, 2 = can move if needed
VISIT_TYPES = {
    'SOC':     (90, 'RN', 1),
    'RECERT':  (60, 'RN', 1),
    'DC':      (45, 'RN', 2),
    'ROUTINE': (None, 'LPN', 2),  # routine length is drawn per patient
}
ROUTINE_TIMES = [30, 45, 60]
ROUTINE_PROBS = [0.30, 0.45, 0.25]
P_NEW_PATIENT = 0.15      # first visit this week is SOC, no prior nurse
P_RECERT = 0.10           # one visit this week is a recert
P_DISCHARGE = 0.08        # last visit this week is a discharge

# arrival windows set by the agency: the nurse must START the visit inside [tw_start, tw_end]
WINDOW_TYPES = {
    'ANY':   ('8:00', '17:00'),
    'AM':    ('8:00', '12:00'),
    'PM':    ('12:00', '17:00'),
    'FIXED': None,  # 1-hour window (e.g. insulin, wound vac change), start drawn per patient
}
WINDOW_PROBS = [0.50, 0.25, 0.15, 0.10]
FIXED_STARTS = ['8:00', '9:00', '10:00', '13:00', '14:00']

CIRCUITY = 1.35      # road miles / straight-line miles (typical urban factor)
AVG_MPH = 22         # average city driving speed
PARKING_MIN = 3      # fixed overhead per trip

TRAVEL_BUFFER = 30   # must match travel_buffer in model.py


def haversine_miles(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 3958.8 * math.asin(math.sqrt(h))


def time_to_minutes(t):
    h, m = map(int, t.split(':'))
    return h * 60 + m


def minutes_to_time(m):
    return f'{m // 60}:{m % 60:02d}'


def can_do(nurse_discipline, required):
    return nurse_discipline == 'RN' or required == 'LPN'


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    # nurses + availability
    avail_rows, nurse_rows = [], []
    for nurse_id, discipline, home_zip, max_visits, shifts in NURSES:
        nurse_rows.append({'nurse_id': nurse_id, 'discipline': discipline, 'home_zip': home_zip,
                           'lat': ZIPS[home_zip][0], 'lon': ZIPS[home_zip][1],
                           'max_visits_per_day': max_visits, 'cost_per_mile': COST_PER_MILE})
        for day in DAYS:
            if day in shifts:
                s, e = shifts[day]
                avail_rows.append({'nurse_id': nurse_id, 'day': day, 'available': 1, 'start_time': s, 'end_time': e})
            else:
                avail_rows.append({'nurse_id': nurse_id, 'day': day, 'available': 0, 'start_time': None, 'end_time': None})
    nurse_avail = pd.DataFrame(avail_rows)
    nurses = pd.DataFrame(nurse_rows)

    # patients + appointments
    zip_codes = list(ZIPS)
    patterns = [p for p, _ in VISIT_PATTERNS]
    pattern_probs = [w for _, w in VISIT_PATTERNS]
    window_names = list(WINDOW_TYPES)

    patient_rows, appt_rows, locations = [], [], {}
    for k in range(1, N_PATIENTS + 1):
        patient_id = f'P{k:03d}'
        z = zip_codes[rng.integers(len(zip_codes))]
        # jitter within ~0.5 mile of the zip centroid so patients in the same zip are distinct
        lat = round(ZIPS[z][0] + rng.normal(0, 0.004), 5)
        lon = round(ZIPS[z][1] + rng.normal(0, 0.005), 5)
        locations[patient_id] = (lat, lon)

        pattern = patterns[rng.choice(len(patterns), p=pattern_probs)]
        visit_days = pattern if pattern is not None else [DAYS[rng.integers(len(DAYS))]]
        routine_time = int(rng.choice(ROUTINE_TIMES, p=ROUTINE_PROBS))

        window_type = window_names[rng.choice(len(window_names), p=WINDOW_PROBS)]
        if window_type == 'FIXED':
            tw_start = FIXED_STARTS[rng.integers(len(FIXED_STARTS))]
            tw_end = minutes_to_time(time_to_minutes(tw_start) + 60)
        else:
            tw_start, tw_end = WINDOW_TYPES[window_type]

        # visit type per day
        is_new = rng.random() < P_NEW_PATIENT
        types = ['ROUTINE'] * len(visit_days)
        if is_new:
            types[0] = 'SOC'
        elif rng.random() < P_RECERT:
            types[rng.integers(len(types))] = 'RECERT'
        elif rng.random() < P_DISCHARGE:
            types[-1] = 'DC'

        # continuity history: existing patients were last seen by some nurse qualified for routine visits
        prior_nurse = None if is_new else NURSES[rng.integers(len(NURSES))][0]

        patient_rows.append({'patient_id': patient_id, 'location': z, 'lat': lat, 'lon': lon,
                             'visits_per_week': len(visit_days), 'window_type': window_type,
                             'tw_start': tw_start, 'tw_end': tw_end, 'prior_nurse_id': prior_nurse})

        for day, visit_type in zip(visit_days, types):
            minutes, discipline, priority = VISIT_TYPES[visit_type]
            appt_rows.append({'patient_id': patient_id, 'day': day, 'location': z, 'lat': lat, 'lon': lon,
                              'service_time': minutes or routine_time, 'visit_type': visit_type,
                              'required_discipline': discipline, 'tw_start': tw_start, 'tw_end': tw_end,
                              'priority': priority})

    patients = pd.DataFrame(patient_rows)
    patient_appt = pd.DataFrame(appt_rows)
    patient_appt['day_idx'] = patient_appt['day'].map(DAYS.index)
    patient_appt = patient_appt.sort_values(['day_idx', 'patient_id']).drop(columns='day_idx').reset_index(drop=True)
    patient_appt.insert(0, 'appointment_id', [f'A{i:03d}' for i in range(1, len(patient_appt) + 1)])

    # travel matrix: every ordered pair of Agency, nurse home bases and patients
    points = {'Agency': ZIPS[AGENCY_ZIP]}
    points.update({r['nurse_id']: (r['lat'], r['lon']) for r in nurse_rows})
    points.update(locations)

    travel_rows = []
    for origin, o_loc in points.items():
        for dest, d_loc in points.items():
            if origin == dest:
                continue
            miles = haversine_miles(o_loc, d_loc) * CIRCUITY
            minutes = miles / AVG_MPH * 60 + PARKING_MIN
            travel_rows.append({'origin': origin, 'destination': dest,
                                'travel_minutes': int(round(minutes)), 'distance_miles': round(miles, 1)})
    dist = pd.DataFrame(travel_rows)
    travel = {(r.origin, r.destination): r.travel_minutes for r in dist.itertuples()}

    # sanity checks
    shifts = {(r.nurse_id, r.day): (time_to_minutes(r.start_time), time_to_minutes(r.end_time))
              for r in nurse_avail.itertuples() if r.available == 1}
    disc = dict(zip(nurses['nurse_id'], nurses['discipline']))
    max_visits = dict(zip(nurses['nurse_id'], nurses['max_visits_per_day']))

    # 1. every appointment has at least one qualified nurse who can reach it inside its window and finish in shift
    for a in patient_appt.itertuples():
        ok = False
        for (n, d), (s, e) in shifts.items():
            if d != a.day or not can_do(disc[n], a.required_discipline):
                continue
            arrive = max(s + travel[(n, a.patient_id)], time_to_minutes(a.tw_start))
            if arrive <= time_to_minutes(a.tw_end) and arrive + a.service_time <= e:
                ok = True
                break
        assert ok, f'{a.appointment_id} ({a.patient_id} {a.day}) cannot be served by any qualified nurse'

    # 2. daily capacity, same logic as model.py, plus RN-only demand vs RN capacity and visit-count caps
    print('day    visits  demand_min  capacity_min  rn_demand  rn_capacity  visit_cap')
    for day in DAYS:
        today = patient_appt[patient_appt['day'] == day]
        demand = (today['service_time'] + TRAVEL_BUFFER).sum()
        rn_demand = (today.loc[today['required_discipline'] == 'RN', 'service_time'] + TRAVEL_BUFFER).sum()
        on = [(n, e - s) for (n, d), (s, e) in shifts.items() if d == day]
        capacity = sum(m for _, m in on)
        rn_capacity = sum(m for n, m in on if disc[n] == 'RN')
        visit_cap = sum(max_visits[n] for n, _ in on)
        print(f'{day:<6} {len(today):>6}  {demand:>10}  {capacity:>12}  {rn_demand:>9}  {rn_capacity:>11}  {visit_cap:>9}')
        assert demand <= capacity, f'{day}: demand {demand} > capacity {capacity}'
        assert rn_demand <= rn_capacity, f'{day}: RN demand {rn_demand} > RN capacity {rn_capacity}'
        assert len(today) <= visit_cap, f'{day}: {len(today)} visits > visit cap {visit_cap}'

        # 3. a full routing exists: greedy earliest-deadline insertion honoring windows, skills, caps, return home
        route = {n: [n, s, e, 0] for (n, d), (s, e) in shifts.items() if d == day}  # location, clock, shift end, visits
        for a in today.assign(te=today['tw_end'].map(time_to_minutes)).sort_values('te').itertuples():
            best = None
            for n, (loc, clock, e, count) in route.items():
                if count >= max_visits[n] or not can_do(disc[n], a.required_discipline):
                    continue
                arrive = max(clock + travel[(loc, a.patient_id)], time_to_minutes(a.tw_start))
                finish = arrive + a.service_time
                if arrive <= time_to_minutes(a.tw_end) and finish + travel[(a.patient_id, n)] <= e:
                    if best is None or finish < best[1]:
                        best = (n, finish)
            assert best is not None, f'{day}: greedy routing could not place {a.appointment_id}'
            route[best[0]][0], route[best[0]][1] = a.patient_id, best[1]
            route[best[0]][3] += 1

    nurse_avail.to_csv(os.path.join(OUT_DIR, 'nurse_availability.csv'), index=False)
    patient_appt.to_csv(os.path.join(OUT_DIR, 'patient_appointments.csv'), index=False)
    dist.to_csv(os.path.join(OUT_DIR, 'travel_matrix.csv'), index=False)
    nurses.to_csv(os.path.join(OUT_DIR, 'nurses.csv'), index=False)
    patients.to_csv(os.path.join(OUT_DIR, 'patients.csv'), index=False)

    print(f'\n{len(nurses)} nurses, {len(patients)} patients, {len(patient_appt)} appointments '
          f'({patient_appt["visit_type"].value_counts().to_dict()}), {len(dist)} travel pairs -> {OUT_DIR}')


if __name__ == '__main__':
    main()
