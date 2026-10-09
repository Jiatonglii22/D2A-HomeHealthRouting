"""
Pull a travel-time / distance matrix from the public OSRM demo server.

Writes travel_car.csv (long format, for Gurobi).

Usage:
    python Datasets/osrm_travel_matrix.py                         # Datasets/Forecast (default)
    python Datasets/osrm_travel_matrix.py 
"""

import argparse
import time

import requests
import pandas as pd
from pathlib import Path

# Coordinates are (latitude, longitude) here -- the script flips them for OSRM.
DEPOT = ("Agency", 40.6847801208496, -80.0629425048828)
DEFAULT_DATA_DIR = Path("Datasets") / "Forecast"
OUTPUT_PATH = Path("Datasets") / "Processed" / "travel_car.csv"

OSRM_URL = "http://router.project-osrm.org/table/v1/driving/"
MAX_COORDS_PER_REQUEST = 100
REQUEST_PAUSE_SECONDS = 1 


def fetch_block(sources, destinations):
    # OSRM wants lon,lat
    block_nodes = sources + destinations
    coord_str = ";".join(f"{lon},{lat}" for _, lat, lon in block_nodes)
    src_idx = ";".join(str(k) for k in range(len(sources)))
    dst_idx = ";".join(str(len(sources) + k) for k in range(len(destinations)))
    url = (
        f"{OSRM_URL}{coord_str}?sources={src_idx}&destinations={dst_idx}"
        "&annotations=duration,distance"
    )

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM error: {data}")

    return data["durations"], data["distances"]


def fetch_matrix(nodes):
    # Split the full N x N table into blocks of at most MAX_COORDS_PER_REQUEST coordinates
    n = len(nodes)
    block = MAX_COORDS_PER_REQUEST // 2
    durations = [[None] * n for _ in range(n)]
    distances = [[None] * n for _ in range(n)]

    starts = range(0, n, block)
    total = len(starts) ** 2
    done = 0
    for si in starts:
        for di in starts:
            dur, dist = fetch_block(nodes[si:si + block], nodes[di:di + block])
            for a, (dur_row, dist_row) in enumerate(zip(dur, dist)):
                durations[si + a][di:di + len(dur_row)] = dur_row
                distances[si + a][di:di + len(dist_row)] = dist_row
            done += 1
            print(f"  block {done}/{total}")
            time.sleep(REQUEST_PAUSE_SECONDS)

    unroutable = [
        (nodes[i][0], nodes[j][0])
        for i in range(n) for j in range(n)
        if durations[i][j] is None or distances[i][j] is None
    ]
    if unroutable:
        raise RuntimeError(f"OSRM found no route for {len(unroutable)} pairs, e.g. {unroutable[:5]}")

    return durations, distances


def load_nodes(data_dir):
    nurses = pd.read_csv(data_dir / "nurses.csv")
    nurse_nodes = nurses[["nurse_id", "lat", "lon"]].itertuples(
        index=False, name=None
    )
    patients = pd.read_csv(data_dir / "patients.csv")
    patient_nodes = patients[["patient_id", "lat", "lon"]].itertuples(
        index=False, name=None
    )
    return [DEPOT, *nurse_nodes, *patient_nodes]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="folder with nurses.csv and patients.csv (default: Datasets/Forecast)")
    args = parser.parse_args()

    nodes = load_nodes(args.data_dir)
    names = [node[0] for node in nodes]
    print(f"Fetching {len(names)} x {len(names)} matrix from OSRM for {args.data_dir}")
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
    df.to_csv(OUTPUT_PATH, index=False)

    # Sanity checks
    diag = [durations[i][i] for i in range(len(names))]
    print(f"\nDiagonal all zero: {all(v == 0 for v in diag)}")
    print(f"Nodes: {len(names)}  |  rows written: {len(df)}  |  {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
