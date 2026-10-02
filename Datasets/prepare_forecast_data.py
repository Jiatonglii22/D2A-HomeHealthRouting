"""
Turn the forecast output of Prediction_Model/12_synthetic_patient_data.py into the
input format main.py / Optimization_Model/model.py expect (same layout as Datasets/Dummy).

Reads   Datasets/Processed/synthetic_patients_<AGENCY>.csv
        Datasets/Processed/synthetic_appointments_<AGENCY>.csv
Writes  Datasets/Forecast/patients.csv
        Datasets/Forecast/patient_appointments.csv
        Datasets/Forecast/nurses.csv              (synthetic roster)
        Datasets/Forecast/nurse_availability.csv  (synthetic shifts)

The forecast has no staffing data, so the nurse roster is generated here (seeded):
  - headcount is sized from the busiest day's workload, plus slack
  - each nurse's home base is placed in a ZIP drawn in proportion to forecast visits,
    next to a forecast patient location in that ZIP (OSRM snaps it to the nearest road)

Next steps:  python Datasets/osrm_travel_matrix.py   then   python main.py
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
AGENCY_ID = "397636"

HERE = Path(__file__).parent
PATIENTS_IN = HERE / "Processed" / f"synthetic_patients_{AGENCY_ID}.csv"
APPOINTMENTS_IN = HERE / "Processed" / f"synthetic_appointments_{AGENCY_ID}.csv"
OUT_DIR = HERE / "Forecast"

# Day labels must match `days` in Optimization_Model/model.py
DAY_MAP = {
    "Monday": "Mon",
    "Tuesday": "Tues",
    "Wednesday": "Wed",
    "Thursday": "Thurs",
    "Friday": "Fri",
}
DAYS = list(DAY_MAP.values())

# --- staffing assumptions -----------------------------------------------------
SHIFT_MINUTES = 480                      # model.py caps shifts at 8 h
SHIFT_STARTS = ["7:00", "7:30", "8:00", "8:30", "9:00"]
ASSUMED_TRAVEL_PER_VISIT = 30            # minutes, for sizing only (real times come from OSRM)
STAFFING_SLACK = 1.3                     # headcount = ceil(peak nurses needed x slack)
DAY_OFF_PROB = 0.25                      # chance a nurse has one weekday off
MIN_DAILY_CAPACITY_RATIO = 1.15          # on-shift capacity / need, every day
HOME_JITTER_DEG = 0.003                  # ~300 m, so homes don't sit on a patient's door
COST_PER_MILE = 0.5                      # same as `cpg` in model.py


def load_patients():
    p = pd.read_csv(PATIENTS_IN, dtype={"zip": str})
    return pd.DataFrame({
        "patient_id": p["patient_id"],
        "location": p["zip"],
        "lat": p["latitude"].round(6),
        "lon": p["longitude"].round(6),
        "visits_per_week": p["weekly_visits"],
        "annual_episodes": p["annual_episodes"],
    })


def load_appointments():
    a = pd.read_csv(APPOINTMENTS_IN, dtype={"zip": str})

    unknown = set(a["day"]) - set(DAY_MAP)
    if unknown:
        raise ValueError(f"Unexpected day labels in {APPOINTMENTS_IN.name}: {unknown}")

    appts = pd.DataFrame({
        "appointment_id": a["appointment_id"],
        "patient_id": a["patient_id"],
        "day": a["day"].map(DAY_MAP),
        "location": a["zip"],
        "lat": a["latitude"].round(6),
        "lon": a["longitude"].round(6),
        "service_time": a["service_time_minutes"],
    })

    # model.py reads one row per (patient, day); a second one would be silently ignored
    dupes = appts.duplicated(["patient_id", "day"])
    if dupes.any():
        raise ValueError(f"{dupes.sum()} patients have more than one visit on the same day")

    day_order = {d: i for i, d in enumerate(DAYS)}
    return appts.sort_values(
        ["day", "appointment_id"], key=lambda s: s.map(day_order) if s.name == "day" else s
    ).reset_index(drop=True)


def daily_nurses_needed(appts):
    workload = (appts["service_time"] + ASSUMED_TRAVEL_PER_VISIT).groupby(appts["day"]).sum()
    return (workload / SHIFT_MINUTES).reindex(DAYS, fill_value=0)


def make_nurses(rng, appts, n_nurses):
    # ZIP weights = forecast visits per ZIP; home = jittered forecast location in that ZIP
    zip_weights = appts["location"].value_counts(normalize=True)
    home_zips = rng.choice(zip_weights.index.to_numpy(), size=n_nurses, p=zip_weights.to_numpy())

    rows = []
    for k, zip_code in enumerate(home_zips, start=1):
        in_zip = appts[appts["location"] == zip_code]
        anchor = in_zip.iloc[rng.integers(len(in_zip))]
        rows.append({
            "nurse_id": f"N{k:03d}",
            "discipline": "RN",
            "home_zip": zip_code,
            "lat": round(anchor["lat"] + rng.uniform(-HOME_JITTER_DEG, HOME_JITTER_DEG), 6),
            "lon": round(anchor["lon"] + rng.uniform(-HOME_JITTER_DEG, HOME_JITTER_DEG), 6),
            "cost_per_mile": COST_PER_MILE,
        })
    return pd.DataFrame(rows)


def make_availability(rng, nurse_ids, need):
    # Redraw days off until every day keeps enough nurses on shift
    for _ in range(1000):
        day_off = {
            n: (rng.choice(DAYS) if rng.random() < DAY_OFF_PROB else None) for n in nurse_ids
        }
        on_shift = pd.Series({d: sum(off != d for off in day_off.values()) for d in DAYS})
        if (on_shift >= need * MIN_DAILY_CAPACITY_RATIO).all():
            break
    else:
        raise RuntimeError("Could not draw days off that leave enough capacity every day")

    rows = []
    for n in nurse_ids:
        start = rng.choice(SHIFT_STARTS)
        h, m = map(int, start.split(":"))
        end_min = h * 60 + m + SHIFT_MINUTES
        end = f"{end_min // 60}:{end_min % 60:02d}"
        for d in DAYS:
            working = day_off[n] != d
            rows.append({
                "nurse_id": n,
                "day": d,
                "available": int(working),
                "start_time": start if working else None,
                "end_time": end if working else None,
            })
    return pd.DataFrame(rows), on_shift


def main():
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    patients = load_patients()
    appts = load_appointments()

    missing = set(appts["patient_id"]) - set(patients["patient_id"])
    if missing:
        raise ValueError(f"{len(missing)} appointment patient_ids are missing from patients file")

    need = daily_nurses_needed(appts)
    n_nurses = math.ceil(need.max() * STAFFING_SLACK)

    nurses = make_nurses(rng, appts, n_nurses)
    availability, on_shift = make_availability(rng, nurses["nurse_id"].tolist(), need)

    patients.to_csv(OUT_DIR / "patients.csv", index=False)
    appts.to_csv(OUT_DIR / "patient_appointments.csv", index=False)
    nurses.to_csv(OUT_DIR / "nurses.csv", index=False)
    availability.to_csv(OUT_DIR / "nurse_availability.csv", index=False)

    summary = pd.DataFrame({
        "visits": appts["day"].value_counts().reindex(DAYS, fill_value=0),
        "nurses_needed": need.round(1),
        "nurses_on_shift": on_shift,
    })
    print(f"Patients: {len(patients)}  |  appointments: {len(appts)}  |  nurses: {n_nurses}")
    print(summary.to_string())
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()
