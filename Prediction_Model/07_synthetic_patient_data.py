import os

import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import Point

SEED = 42
rng = np.random.default_rng(SEED)

AGENCY_ID = "397636"

# Agency location: UPMC Home Healthcare
AGENCY_LATITUDE = 40.6847801208496
AGENCY_LONGITUDE = -80.0629425048828

# Synthetic planning horizon
DAYS = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday"
]

# Average number of weeks in a year
WEEKS_PER_YEAR = 52

# Appointment durations
APPOINTMENT_DURATIONS = [30, 60, 90]

# Visits per patient per week
MIN_VISITS_PER_PATIENT = 1
MAX_VISITS_PER_PATIENT = 3


# file paths 
UTILIZATION_FILE = ("Datasets/Raw/HH_UtilizationZip_2024.csv")
AGENCY_ZIP_FILE = ("Datasets/Processed/agency_zip_demand.csv")
ZIP_SHAPEFILE = ("Datasets/Raw/AlleghenyCounty__ZipCodeBoundaries202607.shp")
PATIENT_OUTPUT = ("Datasets/Processed/synthetic_patients_397636.csv")
APPOINTMENT_OUTPUT = ("Datasets/Processed/synthetic_appointments_397636.csv")


# create output directory 
os.makedirs(
    "Datasets/Processed",
    exist_ok=True
)

# 1. LOAD CMS UTILIZATION
print("Loading CMS utilization data...")

cms = pd.read_csv(
    UTILIZATION_FILE,
    low_memory=False
)

cms["PRVDR_ID"] = (
    pd.to_numeric(
        cms["PRVDR_ID"],
        errors="coerce"
    )
    .astype("Int64")
    .astype(str)
)

# Remove possible "<NA>" strings
cms = cms[cms["PRVDR_ID"] != "<NA>"].copy()


# 2. FILTER CMS DATA
# Keep:
#   2024
#   Calendar year
#   Home Health
#   Selected agency

agency_cms = cms[
    (cms["YEAR"] == 2024)
    &
    (cms["YEAR_TYPE"] == "CY")
    &
    (cms["SRVC_CTGRY"] == "HH")
    &
    (cms["PRVDR_ID"] == AGENCY_ID)
].copy()

if agency_cms.empty:
    raise ValueError(
        f"No 2024 CMS utilization record found "
        f"for agency {AGENCY_ID}."
    )

# 3. GET ANNUAL UTILIZATION
def numeric_value(df, column):
    """
    Convert a CMS numeric column to a usable number.
    """
    return (
        pd.to_numeric(
            df[column],
            errors="coerce"
        )
        .fillna(0)
        .sum()
    )

annual_beneficiaries = int(
    numeric_value(
        agency_cms,
        "BENE_DSTNCT_CNT"
    )
)

annual_nursing_visits = int(
    numeric_value(
        agency_cms,
        "NRSNG_VISITS_CNT"
    )
)

print("\nAnnual CMS utilization:")
print(f"  Beneficiaries:  {annual_beneficiaries:,}")
print(f"  Nursing visits: {annual_nursing_visits:,}")

# 4. CONVERT ANNUAL NURSING VISITS TO WEEKLY DEMAND
expected_weekly_visits = (
    annual_nursing_visits
    / WEEKS_PER_YEAR
)

print("\nExpected weekly utilization:")
print(
    f"  Weekly nursing visits: "
    f"{expected_weekly_visits:.2f}"
)

# 5. SAMPLE A REALISTIC WEEKLY NUMBER OF VISITS
# Use a Poisson distribution around the annual average.
weekly_nursing_visits = int(
    rng.poisson(
        expected_weekly_visits
    )
)

print(
    "\nSampled nursing visits for this "
    "synthetic week: "
    f"{weekly_nursing_visits}"
)

# 6. LOAD AGENCY × ZIP DEMAND
print(
    "\nLoading agency × ZIP demand..."
)

agency_zip = pd.read_csv(
    AGENCY_ZIP_FILE
)

agency_zip["PRVDR_ID"] = (
    pd.to_numeric(
        agency_zip["PRVDR_ID"],
        errors="coerce"
    )
    .astype("Int64")
    .astype(str)
)

agency_zip = agency_zip[
    agency_zip["PRVDR_ID"] == AGENCY_ID
].copy()

if agency_zip.empty:
    raise ValueError(
        f"No agency-ZIP demand found "
        f"for {AGENCY_ID}."
    )

# 7. CHECK FOR NURSING-VISIT DEMAND COLUMN
required_demand_column = (
    "estimated_agency_zip_nursing_visits"
)

if required_demand_column not in agency_zip.columns:
    raise ValueError(
        f"Column '{required_demand_column}' "
        f"was not found in agency_zip_demand.csv.\n"
        f"Available columns:\n"
        f"{agency_zip.columns.tolist()}"
    )

# 8. LOAD ZIP POLYGONS
print("\nLoading ZIP polygons...")

zip_gdf = gpd.read_file(ZIP_SHAPEFILE)

zip_gdf["ZIP"] = (
    zip_gdf["ZIP"]
    .astype(str)
    .str.strip()
)

agency_zip["zip"] = (
    agency_zip["zip"]
    .astype(str)
    .str.strip()
)

# Keep only ZIPs for this agency that have geometry
agency_zip = agency_zip[
    agency_zip["zip"].isin(
        zip_gdf["ZIP"]
    )
].copy()

if agency_zip.empty:
    raise ValueError(
        "No agency ZIPs matched "
        "the ZIP shapefile."
    )

print(
    f"Matched {len(agency_zip)} "
    f"agency-ZIP rows to ZIP polygons."
)

# 9. CREATE ZIP SAMPLING PROBABILITIES
# estimated_agency_zip_nursing_visits represents the
# estimated annual nursing visits for this agency in
# each ZIP.
# We use this as the probability that a synthetic
# nursing visit/patient is located in each ZIP.

total_zip_demand = (
    agency_zip[
        required_demand_column
    ].sum()
)

if total_zip_demand <= 0:
    raise ValueError(
        "Total estimated ZIP nursing-visit "
        "demand must be greater than zero."
    )

agency_zip["zip_probability"] = (
    agency_zip[
        required_demand_column
    ]
    / total_zip_demand
)

# Numerical cleanup
agency_zip["zip_probability"] = (
    agency_zip["zip_probability"]
    .fillna(0)
)

# Normalize again to ensure probabilities
# sum exactly to 1
agency_zip["zip_probability"] = (
    agency_zip["zip_probability"]
    / agency_zip["zip_probability"].sum()
)

print("\nZIP visit-demand probabilities:")

print(
    agency_zip[
        [
            "zip",
            required_demand_column,
            "zip_probability"
        ]
    ]
    .sort_values(
        "zip_probability",
        ascending=False
    )
    .head(20)
)


# 10. CREATE SYNTHETIC BENEFICIARIES
print("\nCreating synthetic beneficiaries...")

patients = pd.DataFrame({
    "patient_id": [
        f"P{i:06d}"
        for i in range(
            1,
            annual_beneficiaries + 1
        )
    ]
})

patients["agency_id"] = AGENCY_ID

# 11. DETERMINE WEEKLY VISIT PARTICIPANTS
# Each active patient receives between 1 and 3
# nursing visits during the synthetic week.

max_possible_visits = (
    len(patients)
    * MAX_VISITS_PER_PATIENT
)

if weekly_nursing_visits > max_possible_visits:
    raise ValueError(
        f"Weekly demand ({weekly_nursing_visits}) "
        f"exceeds the maximum possible visits "
        f"({max_possible_visits}) with at most "
        f"{MAX_VISITS_PER_PATIENT} visits per patient."
    )

if weekly_nursing_visits < len(patients):
    # If weekly demand is smaller than the total
    # number of beneficiaries, only some patients
    # can receive visits during this synthetic week.
    min_active_patients = 1

    max_active_patients = (weekly_nursing_visits)

else:
    # If there are enough visits for every patient
    # to receive at least one visit.
    min_active_patients = int(
        np.ceil(
            weekly_nursing_visits
            / MAX_VISITS_PER_PATIENT
        )
    )

    max_active_patients = min(
        weekly_nursing_visits,
        len(patients)
    )

# Randomly choose the number of patients who receive visits
n_active_patients = int(
    rng.integers(
        min_active_patients,
        max_active_patients + 1
    )
)

# Randomly select the patients who will have visits this week
active_patient_indices = rng.choice(
    patients.index,
    size=n_active_patients,
    replace=False
)

weekly_patients = (
    patients
    .loc[active_patient_indices]
    .copy()
)

# 12. ASSIGN 1–3 WEEKLY VISITS
# Start every active patient with 1 visit.
weekly_patients["weekly_visits"] = 1

# Remaining visits
remaining_visits = (
    weekly_nursing_visits
    - n_active_patients
)

while remaining_visits > 0:
    # Patients who can still receive another visit
    eligible_indices = (
        weekly_patients.index[
            weekly_patients[
                "weekly_visits"
            ]
            < MAX_VISITS_PER_PATIENT
        ]
    )

    if len(eligible_indices) == 0:
        raise ValueError(
            "Unable to allocate all weekly "
            "nursing visits."
        )

    # Randomly choose one eligible patient
    patient_idx = rng.choice(
        eligible_indices
    )

    # Add one visit
    weekly_patients.loc[
        patient_idx,
        "weekly_visits"
    ] += 1

    remaining_visits -= 1

# 13. PRINT VISIT DISTRIBUTION
patients_1_visit = (
    weekly_patients[
        "weekly_visits"
    ] == 1
).sum()

patients_2_visits = (
    weekly_patients[
        "weekly_visits"
    ] == 2
).sum()

patients_3_visits = (
    weekly_patients[
        "weekly_visits"
    ] == 3
).sum()


print("\nWeekly visit distribution:")

print(
    f"  1 visit/week: "
    f"{patients_1_visit} patients"
)

print(
    f"  2 visits/week: "
    f"{patients_2_visits} patients"
)

print(
    f"  3 visits/week: "
    f"{patients_3_visits} patients"
)

print(
    f"  Total patients: "
    f"{len(weekly_patients)}"
)

print(
    f"  Total visits: "
    f"{weekly_patients['weekly_visits'].sum()}"
)

print(
    f"  Target weekly visits: "
    f"{weekly_nursing_visits}"
)

# 14. ASSIGN ZIP CODES TO WEEKLY PATIENTS
# ZIP assignment is based on the agency's estimated
# annual nursing-visit demand by ZIP.

weekly_patients["zip"] = rng.choice(
    agency_zip["zip"].to_numpy(),
    size=len(weekly_patients),
    p=agency_zip["zip_probability"].to_numpy()
)

# 15. GENERATE RANDOM LOCATION INSIDE EACH ZIP POLYGON
print("\nGenerating patient locations...")

# Use same CRS as shapefile for point generation
working_crs = zip_gdf.crs
patient_records = []

for _, patient in weekly_patients.iterrows():
    zip_code = patient["zip"]

    polygon = zip_gdf.loc[
        zip_gdf["ZIP"] == zip_code,
        "geometry"
    ].iloc[0]

    # Bounding box
    minx, miny, maxx, maxy = (polygon.bounds)

    # Rejection sampling
    point = None

    for _ in range(10000):

        x = rng.uniform(minx, maxx)

        y = rng.uniform(miny, maxy)

        candidate = Point(x, y)

        if polygon.contains(candidate):
            point = candidate
            break

    if point is None:
        # Fall back to representative point
        point = polygon.representative_point()

    patient_records.append({
        "patient_id": patient["patient_id"],
        "agency_id": AGENCY_ID,
        "zip": zip_code,
        "weekly_visits": patient["weekly_visits"],
        "geometry": point
    })


patient_gdf = gpd.GeoDataFrame(
    patient_records,
    geometry="geometry",
    crs=working_crs
)

# 16. CONVERT TO LATITUDE/LONGITUDE
patient_gdf = patient_gdf.to_crs("EPSG:4326")

patient_gdf["longitude"] = (patient_gdf.geometry.x)

patient_gdf["latitude"] = (patient_gdf.geometry.y)


# Drop geometry before saving CSV
patients_output = patient_gdf.drop(columns="geometry")

# 17. SAVE PATIENT DATA
patients_output.to_csv(
    PATIENT_OUTPUT,
    index=False
)

print(
    f"\nSaved patient data to:\n"
    f"{PATIENT_OUTPUT}"
)

# 18. CREATE WEEKLY APPOINTMENTS
# Each nursing visit becomes one appointment.
# Each patient has 1–3 appointments per week.
# A patient cannot have more than one appointment
# on the same weekday.
# Appointment durations:
#     30 minutes
#     60 minutes
#     90 minutes

appointments = []

for _, patient in patients_output.iterrows():

    n_visits = int(patient["weekly_visits"])

    # Randomly select distinct weekdays
    selected_days = rng.choice(
        DAYS,
        size=n_visits,
        replace=False
    )

    # Sort weekdays chronologically
    day_order = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4
    }

    selected_days = sorted(
        selected_days,
        key=lambda x: day_order[x]
    )

    for visit_number, day in enumerate(
        selected_days,
        start=1
    ):

        # Randomly select appointment duration
        service_time = int(
            rng.choice(
                APPOINTMENT_DURATIONS
            )
        )

        appointments.append({
            "appointment_id": (
                f"{patient['patient_id']}_"
                f"V{visit_number}"
            ),

            "patient_id": patient["patient_id"],
            "agency_id": AGENCY_ID,
            "zip": patient["zip"],
            "day": day,
            "service_time_minutes": service_time,
            "latitude": patient["latitude"],
            "longitude": patient["longitude"]
        })


appointments_df = pd.DataFrame(appointments)

# 19. SAVE APPOINTMENTS
appointments_df.to_csv(
    APPOINTMENT_OUTPUT,
    index=False
)

print(
    f"Saved appointments to:\n"
    f"{APPOINTMENT_OUTPUT}"
)

# 20. VALIDATION
print("\n" + "=" * 60)
print("VALIDATION")
print("=" * 60)

print(
    f"Annual beneficiaries: "
    f"{annual_beneficiaries:,}"
)

print(
    f"Annual nursing visits: "
    f"{annual_nursing_visits:,}"
)

print(
    f"Expected weekly visits: "
    f"{expected_weekly_visits:.2f}"
)

print(
    f"Sampled CMS weekly visits: "
    f"{weekly_nursing_visits:,}"
)

print(
    f"Generated appointments: "
    f"{len(appointments_df):,}"
)

print(
    f"Weekly patients: "
    f"{len(weekly_patients):,}"
)

print(
    f"Minimum visits/patient: "
    f"{appointments_df.groupby('patient_id').size().min()}"
)

print(
    f"Maximum visits/patient: "
    f"{appointments_df.groupby('patient_id').size().max()}"
)


print("\nVisits by weekday:")

print(
    appointments_df["day"]
    .value_counts()
    .reindex(DAYS)
    .fillna(0)
    .astype(int)
)

print("\nAppointment duration distribution:")

print(
    appointments_df[
        "service_time_minutes"
    ]
    .value_counts()
    .sort_index()
)

print("\nAppointments by ZIP:")

print(
    appointments_df["zip"]
    .value_counts()
    .head(10)
)

print("\nLocation summary:")

print(
    appointments_df[
        [
            "latitude",
            "longitude"
        ]
    ].describe()
)

# 21. FINAL VALIDATION CHECKS
# Total appointments must exactly equal the
# sampled weekly nursing-visit target.

assert (
    len(appointments_df)
    == weekly_nursing_visits
), (
    "Total generated appointments do not match "
    "the CMS-derived weekly nursing-visit target."
)

# Every patient has between 1 and 3 appointments.

visits_per_patient = (
    appointments_df
    .groupby("patient_id")
    .size()
)

assert (
    visits_per_patient.between(
        1,
        3
    ).all()
), (
    "Some patients have fewer than 1 or "
    "more than 3 appointments."
)

# No patient has multiple appointments
# on the same weekday.

assert not appointments_df.duplicated(
    subset=[
        "patient_id",
        "day"
    ]
).any(), (
    "A patient has multiple appointments "
    "on the same day."
)


# Every appointment has a valid duration.
assert appointments_df[
    "service_time_minutes"
].isin(
    APPOINTMENT_DURATIONS
).all(), (
    "Unexpected appointment duration found."
)

print(
    "\nAll appointment validation checks passed."
)

print("\nDone.")