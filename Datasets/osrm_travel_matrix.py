"""
Pull a travel-time / distance matrix from the public OSRM demo server.

Writes travel_car.csv (long format, for Gurobi) plus two wide matrices for eyeballing.
"""

import requests
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Coordinates are (latitude, longitude) here -- the script flips them for OSRM.
# ---------------------------------------------------------------------------
DEPOT = ("Agency", 40.6847801208496, -80.0629425048828)
NURSES_CSV = Path(__file__).parent / "Dummy" / "nurses.csv"
PATIENTS_CSV = Path(__file__).parent / "Dummy" / "patients.csv"

OSRM_URL = "http://router.project-osrm.org/table/v1/driving/"


def fetch_matrix(nodes):
    # OSRM wants lon,lat
    coord_str = ";".join(f"{lon},{lat}" for _, lat, lon in nodes)
    url = f"{OSRM_URL}{coord_str}?annotations=duration,distance"

    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM error: {data}")

    return data["durations"], data["distances"]


def load_nodes():
    nurses = pd.read_csv(NURSES_CSV)
    nurse_nodes = nurses[["nurse_id", "lat", "lon"]].itertuples(
        index=False, name=None
    )
    patients = pd.read_csv(PATIENTS_CSV)
    patient_nodes = patients[["patient_id", "lat", "lon"]].itertuples(
        index=False, name=None
    )
    return [DEPOT, *nurse_nodes, *patient_nodes]


def main():
    nodes = load_nodes()
    names = [node[0] for node in nodes]
    durations, distances = fetch_matrix(nodes)

    rows = []
    for i, from_name in enumerate(names):
        for j, to_name in enumerate(names):
            rows.append({
                "origin": from_name,
                "destination": to_name,
                "travel_minutes": round(durations[i][j] / 60.0, 2),   # OSRM returns seconds
                "distance_miles": round(distances[i][j] / 1609.34, 3),  # OSRM returns meters
            })

    df = pd.DataFrame(rows)
    output_path = Path(__file__).parent / "Processed" / "travel_car.csv"
    df.to_csv(output_path, index=False)

    # Sanity checks
    diag = [durations[i][i] for i in range(len(names))]
    print(f"\nDiagonal (should all be 0): {diag}")
    print(f"Nodes: {len(names)}  |  rows written: {len(df)}")


if __name__ == "__main__":
    main()