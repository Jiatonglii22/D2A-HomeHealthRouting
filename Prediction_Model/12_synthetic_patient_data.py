# Prediction_Model/12_synthetic_patient_data.py

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


# ============================================================
# SETTINGS
# ============================================================

SEED = 42
rng = np.random.default_rng(SEED)

AGENCY_ID = "397636"

# Agency location: UPMC Home Healthcare
AGENCY_LATITUDE = 40.6847801208496
AGENCY_LONGITUDE = -80.0629425048828

# Synthetic planning horizon
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]

# Average number of weeks in a year
WEEKS_PER_YEAR = 52

# Assumed nursing visit duration
SERVICE_TIME_MINUTES = 60


# ============================================================
# FILE PATHS
# ============================================================

UTILIZATION_FILE = (
    "Datasets/Raw/HH_UtilizationZip_2024.csv"
)

AGENCY_ZIP_FILE = (
    "Datasets/Processed/agency_zip_visit_demand.csv"
)

ZIP_SHAPEFILE = (
    "Datasets/Raw/AlleghenyCounty__ZipCodeBoundaries202607.shp"
)

PATIENT_OUTPUT = (
    "Datasets/Processed/synthetic_patients_397636.csv"
)

APPOINTMENT_OUTPUT = (
    "Datasets/Processed/synthetic_appointments_397636.csv"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs("Datasets/Processed", exist_ok=True)


# ============================================================
# 1. LOAD CMS UTILIZATION
# ============================================================

print("Loading CMS utilization data...")

cms = pd.read_csv(UTILIZATION_FILE, low_memory=False)

cms["PRVDR_ID"] = (
    pd.to_numeric(cms["PRVDR_ID"], errors="coerce")
    .astype("Int64")
    .astype(str)
)

# Remove possible "<NA>" strings
cms = cms[cms["PRVDR_ID"] != "<NA>"].copy()

# Filter to:
#   2024
#   Calendar year
#   Home Health
agency_cms = cms[
    (cms["YEAR"] == 2024)
    & (cms["YEAR_TYPE"] == "CY")
    & (cms["SRVC_CTGRY"] == "HH")
    & (cms["PRVDR_ID"] == AGENCY_ID)
].copy()


if agency_cms.empty:
    raise ValueError(
        f"No 2024 CMS utilization record found for agency {AGENCY_ID}."
    )


# ============================================================
# 2. GET ANNUAL UTILIZATION NUMBERS
# ============================================================

def numeric_value(df, column):
    """Convert a CMS numeric column to a usable number."""
    return pd.to_numeric(
        df[column],
        errors="coerce"
    ).fillna(0).sum()


annual_beneficiaries = int(
    numeric_value(
        agency_cms,
        "BENE_DSTNCT_CNT"
    )
)

annual_episodes = int(
    numeric_value(
        agency_cms,
        "TOT_EPSD_STAY_CNT"
    )
)

annual_nursing_visits = int(
    numeric_value(
        agency_cms,
        "NRSNG_VISITS_CNT"
    )
)


print("\nAnnual CMS utilization:")
print(f"  Beneficiaries:    {annual_beneficiaries:,}")
print(f"  Episodes:         {annual_episodes:,}")
print(f"  Nursing visits:   {annual_nursing_visits:,}")


# ============================================================
# 3. CONVERT ANNUAL VISITS TO WEEKLY DEMAND
# ============================================================

expected_weekly_visits = (
    annual_nursing_visits / WEEKS_PER_YEAR
)

expected_weekly_episodes = (
    annual_episodes / WEEKS_PER_YEAR
)


print("\nExpected weekly utilization:")
print(
    f"  Weekly episodes: "
    f"{expected_weekly_episodes:.2f}"
)

print(
    f"  Weekly nursing visits: "
    f"{expected_weekly_visits:.2f}"
)


# ============================================================
# 4. SAMPLE A REALISTIC WEEKLY NUMBER OF VISITS
# ============================================================
#
# We use a Poisson distribution around the annual average.
#
# Example:
#     annual visits = 9,946
#     expected weekly = 191.27
#
# One synthetic week could therefore have:
#     188 visits
#     193 visits
#     197 visits
#     etc.
#
# This avoids artificially forcing every week to contain
# exactly 191 visits.
# ============================================================

weekly_nursing_visits = int(
    rng.poisson(expected_weekly_visits)
)

print(
    f"\nSampled nursing visits for this synthetic week: "
    f"{weekly_nursing_visits}"
)


# ============================================================
# 5. LOAD AGENCY × ZIP DEMAND
# ============================================================

print("\nLoading agency × ZIP demand...")

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
        f"No agency-ZIP demand found for {AGENCY_ID}."
    )


# ============================================================
# 6. LOAD ZIP POLYGONS
# ============================================================

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
    agency_zip["zip"].isin(zip_gdf["ZIP"])
].copy()


if agency_zip.empty:
    raise ValueError(
        "No agency ZIPs matched the ZIP shapefile."
    )


print(
    f"Matched {len(agency_zip)} agency-ZIP rows "
    f"to ZIP polygons."
)


# ============================================================
# 7. CREATE ZIP SAMPLING PROBABILITIES
# ============================================================
#
# estimated_agency_zip_episodes represents the estimated
# annual number of episodes for this agency in each ZIP.
#
# We use this as the probability that a synthetic patient/
# episode is located in each ZIP.
# ============================================================

agency_zip["zip_probability"] = (
    agency_zip["estimated_agency_zip_episodes"]
    / agency_zip["estimated_agency_zip_episodes"].sum()
)

# Numerical cleanup
agency_zip["zip_probability"] = (
    agency_zip["zip_probability"]
    .fillna(0)
)

# Normalize again to ensure probabilities sum to exactly 1
agency_zip["zip_probability"] = (
    agency_zip["zip_probability"]
    / agency_zip["zip_probability"].sum()
)


# ============================================================
# 8. CREATE SYNTHETIC BENEFICIARIES
# ============================================================
#
# We create the actual number of distinct beneficiaries:
#
#       2,257
#
# This is different from the 3,919 annual episodes.
#
# A beneficiary may have more than one episode.
# ============================================================

print("\nCreating synthetic beneficiaries...")

patients = pd.DataFrame({
    "patient_id": [
        f"P{i:06d}"
        for i in range(1, annual_beneficiaries + 1)
    ]
})

patients["agency_id"] = AGENCY_ID


# ============================================================
# 9. ASSIGN ANNUAL EPISODES TO BENEFICIARIES
# ============================================================
#
# Every beneficiary gets at least one episode.
#
# Remaining episodes are randomly assigned to beneficiaries.
#
# Example:
#
#   2,257 beneficiaries
#   3,919 episodes
#
# Therefore:
#
#   3,919 - 2,257 = 1,662
#
# additional episodes are distributed among patients.
# ============================================================

patients["annual_episodes"] = 1

additional_episodes = (
    annual_episodes - annual_beneficiaries
)

if additional_episodes < 0:
    raise ValueError(
        "Annual episodes cannot be less than distinct beneficiaries."
    )


# Probability weighting:
# patients with more episodes become more likely to receive
# another episode. Initially everyone has equal probability.
#
# This creates a mild concentration of repeat episodes.
for _ in range(additional_episodes):

    probabilities = (
        patients["annual_episodes"].to_numpy()
        + 1
    )

    probabilities = (
        probabilities
        / probabilities.sum()
    )

    selected = rng.choice(
        len(patients),
        p=probabilities
    )

    patients.loc[
        selected,
        "annual_episodes"
    ] += 1


print(
    "\nAnnual episode distribution:"
)

print(
    patients["annual_episodes"]
    .value_counts()
    .sort_index()
)


# ============================================================
# 10. SAMPLE WHICH PATIENTS HAVE VISITS THIS WEEK
# ============================================================
#
# A patient's probability of appearing in the week is based
# on their annual number of episodes.
#
# More annual episodes -> higher probability of appearing
# in a randomly selected week.
#
# We sample patients WITHOUT replacement here so that each
# selected patient initially contributes one weekly visit.
# Additional weekly visits are assigned afterward.
# ============================================================

patient_weights = (
    patients["annual_episodes"]
    / patients["annual_episodes"].sum()
)

patient_weights = patient_weights.to_numpy()

# Number of patients needed cannot exceed population
initial_weekly_patients = min(
    weekly_nursing_visits,
    len(patients)
)

selected_patient_indices = rng.choice(
    len(patients),
    size=initial_weekly_patients,
    replace=False,
    p=patient_weights
)

weekly_patients = patients.iloc[
    selected_patient_indices
].copy()

weekly_patients["weekly_visits"] = 1


# ============================================================
# 11. ASSIGN REMAINING WEEKLY VISITS
# ============================================================
#
# If weekly visits > number of selected patients, some patients
# receive a second visit.
#
# The probability is based on annual episode count.
#
# We cap weekly visits at 5 per patient because the current
# planning horizon is Monday-Friday and we want at most one
# visit per patient per day.
# ============================================================

remaining_visits = (
    weekly_nursing_visits
    - len(weekly_patients)
)

MAX_WEEKLY_VISITS_PER_PATIENT = 5


while remaining_visits > 0:

    eligible = weekly_patients[
        weekly_patients["weekly_visits"]
        < MAX_WEEKLY_VISITS_PER_PATIENT
    ].copy()

    if eligible.empty:
        raise ValueError(
            "Unable to assign all weekly visits without "
            "exceeding 5 visits per patient."
        )

    weights = (
        eligible["annual_episodes"]
        .to_numpy()
    )

    weights = weights / weights.sum()

    selected_index = rng.choice(
        eligible.index.to_numpy(),
        p=weights
    )

    weekly_patients.loc[
        selected_index,
        "weekly_visits"
    ] += 1

    remaining_visits -= 1


print(
    "\nWeekly patient/visit summary:"
)

print(
    f"  Patients with visits: "
    f"{len(weekly_patients)}"
)

print(
    f"  Weekly visits: "
    f"{weekly_patients['weekly_visits'].sum()}"
)

print(
    f"  Maximum visits for one patient: "
    f"{weekly_patients['weekly_visits'].max()}"
)


# ============================================================
# 12. ASSIGN ZIP CODES TO WEEKLY PATIENTS
# ============================================================
#
# ZIP assignment is based on the agency's estimated annual
# episode demand by ZIP.
# ============================================================

weekly_patients["zip"] = rng.choice(
    agency_zip["zip"].to_numpy(),
    size=len(weekly_patients),
    p=agency_zip["zip_probability"].to_numpy()
)


# ============================================================
# 13. GENERATE RANDOM LOCATION INSIDE EACH ZIP POLYGON
# ============================================================

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
    minx, miny, maxx, maxy = polygon.bounds

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
        "annual_episodes": patient["annual_episodes"],
        "weekly_visits": patient["weekly_visits"],
        "geometry": point
    })


patient_gdf = gpd.GeoDataFrame(
    patient_records,
    geometry="geometry",
    crs=working_crs
)


# ============================================================
# 14. CONVERT TO LATITUDE/LONGITUDE
# ============================================================

patient_gdf = patient_gdf.to_crs(
    "EPSG:4326"
)

patient_gdf["longitude"] = (
    patient_gdf.geometry.x
)

patient_gdf["latitude"] = (
    patient_gdf.geometry.y
)

# Drop geometry before saving CSV
patients_output = patient_gdf.drop(
    columns="geometry"
)


# ============================================================
# 15. SAVE PATIENT DATA
# ============================================================

patients_output.to_csv(
    PATIENT_OUTPUT,
    index=False
)

print(
    f"\nSaved patient data to:\n"
    f"{PATIENT_OUTPUT}"
)


# ============================================================
# 16. CREATE WEEKLY APPOINTMENTS
# ============================================================
#
# Each nursing visit becomes one appointment.
#
# A patient can have multiple visits in the week, but never
# more than one visit on the same weekday.
# ============================================================

appointments = []

for _, patient in patients_output.iterrows():

    n_visits = int(
        patient["weekly_visits"]
    )

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

        appointments.append({
            "appointment_id": (
                f"{patient['patient_id']}_"
                f"V{visit_number}"
            ),
            "patient_id": patient["patient_id"],
            "agency_id": AGENCY_ID,
            "zip": patient["zip"],
            "day": day,
            "service_time_minutes": SERVICE_TIME_MINUTES,
            "latitude": patient["latitude"],
            "longitude": patient["longitude"]
        })


appointments_df = pd.DataFrame(
    appointments
)


# ============================================================
# 17. SAVE APPOINTMENTS
# ============================================================

appointments_df.to_csv(
    APPOINTMENT_OUTPUT,
    index=False
)

print(
    f"Saved appointments to:\n"
    f"{APPOINTMENT_OUTPUT}"
)


# ============================================================
# 18. VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("VALIDATION")
print("=" * 60)

print(
    f"Annual beneficiaries: "
    f"{annual_beneficiaries:,}"
)

print(
    f"Annual episodes: "
    f"{annual_episodes:,}"
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
    f"Sampled weekly visits: "
    f"{len(appointments_df):,}"
)

print(
    f"Weekly patients: "
    f"{len(weekly_patients):,}"
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

print("\nAppointments by ZIP:")

print(
    appointments_df["zip"]
    .value_counts()
    .head(10)
)

print("\nLocation summary:")

print(
    appointments_df[
        ["latitude", "longitude"]
    ].describe()
)

print("\nDone.")