import pandas as pd
import numpy as np
import os


# ============================================================
# SETTINGS
# ============================================================

PATIENT_FILE = "Datasets/Processed/synthetic_patients_397636.csv"

OUTPUT_FILE = (
    "Datasets/Processed/travel_matrix_397636_100.csv"
)

NUMBER_OF_PATIENTS = 100

# UPMC Home Healthcare
AGENCY_ID = "AGENCY"
AGENCY_LATITUDE = 40.6847801208496
AGENCY_LONGITUDE = -80.0629425048828

# Assumed average driving speed
AVERAGE_SPEED_MPH = 30


# ============================================================
# HAVERSINE DISTANCE
# ============================================================

def haversine_distance(
    lat1,
    lon1,
    lat2,
    lon2
):
    """
    Calculate straight-line distance between
    two latitude/longitude points in miles.
    """

    earth_radius_miles = 3958.8

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    delta_lat = np.radians(lat2 - lat1)
    delta_lon = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(delta_lon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return earth_radius_miles * c


# ============================================================
# LOAD PATIENT DATA
# ============================================================

patients = pd.read_csv(PATIENT_FILE)

print(
    f"Total patients available: {len(patients)}"
)


# ============================================================
# SELECT FIRST 100 PATIENTS
# ============================================================

patients = patients.head(NUMBER_OF_PATIENTS).copy()

print(
    f"Patients used for travel matrix: {len(patients)}"
)


# ============================================================
# CREATE LOCATION TABLE
# ============================================================

locations = []

# Add agency
locations.append(
    {
        "location_id": AGENCY_ID,
        "latitude": AGENCY_LATITUDE,
        "longitude": AGENCY_LONGITUDE,
    }
)

# Add patients
for _, patient in patients.iterrows():

    locations.append(
        {
            "location_id": patient["patient_id"],
            "latitude": patient["latitude"],
            "longitude": patient["longitude"],
        }
    )


locations = pd.DataFrame(locations)


# ============================================================
# CREATE TRAVEL MATRIX
# ============================================================

travel_records = []

for _, origin in locations.iterrows():

    for _, destination in locations.iterrows():

        origin_id = origin["location_id"]
        destination_id = destination["location_id"]

        # Don't calculate travel from a location to itself
        if origin_id == destination_id:
            continue

        distance = haversine_distance(
            origin["latitude"],
            origin["longitude"],
            destination["latitude"],
            destination["longitude"],
        )

        # Estimate travel time
        travel_minutes = (
            distance / AVERAGE_SPEED_MPH
        ) * 60

        travel_records.append(
            {
                "origin": origin_id,
                "destination": destination_id,
                "distance_miles": distance,
                "travel_minutes": travel_minutes,
            }
        )


travel_matrix = pd.DataFrame(
    travel_records
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "Datasets/Processed",
    exist_ok=True
)

travel_matrix.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\nTravel matrix created successfully.")

print(
    f"Number of locations: {len(locations)}"
)

print(
    f"Number of origin-destination pairs: "
    f"{len(travel_matrix)}"
)

print(
    f"\nOutput saved to:\n{OUTPUT_FILE}"
)

print("\nFirst 10 rows:")
print(
    travel_matrix.head(10)
)