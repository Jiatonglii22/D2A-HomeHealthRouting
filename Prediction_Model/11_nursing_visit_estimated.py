import pandas as pd


# =========================================================
# 1. File paths
# =========================================================

demand_file = (
    "Datasets/Processed/"
    "agency_zip_demand.csv"
)

utilization_file = (
    "Datasets/Processed/"
    "cms_agency_utilization_summary.csv"
)

output_file = (
    "Datasets/Processed/"
    "agency_zip_visit_demand.csv"
)


# =========================================================
# 2. Load datasets
# =========================================================

demand = pd.read_csv(demand_file)

utilization = pd.read_csv(utilization_file)

print("Agency-ZIP demand shape:", demand.shape)
print("Agency utilization shape:", utilization.shape)


# =========================================================
# 3. Standardize PRVDR_ID
# =========================================================

# PRVDR_ID is an identifier, not a numerical measurement.
# Convert both datasets to strings so the merge works.

demand["PRVDR_ID"] = (
    demand["PRVDR_ID"]
    .astype(str)
    .str.strip()
)

utilization["PRVDR_ID"] = (
    utilization["PRVDR_ID"]
    .astype(str)
    .str.strip()
)


# =========================================================
# 4. Calculate overall fallback ratio
# =========================================================

# Median agency-level nursing visits per episode.

fallback_ratio = (
    utilization[
        "nursing_visits_per_episode"
    ]
    .median()
)

print("\nFallback nursing visits per episode:")
print(f"{fallback_ratio:.4f}")


# =========================================================
# 5. Keep relevant agency utilization information
# =========================================================

agency_ratios = utilization[
    [
        "PRVDR_ID",
        "nursing_visits_per_episode",
        "agency_episodes"
    ]
].copy()


# =========================================================
# 6. Check agency IDs before merging
# =========================================================

demand_agencies = set(
    demand["PRVDR_ID"]
)

utilization_agencies = set(
    agency_ratios["PRVDR_ID"]
)

matching_agencies = (
    demand_agencies
    & utilization_agencies
)

print("\nAgency ID check:")
print("Agencies in demand data:", len(demand_agencies))
print("Agencies in utilization data:", len(utilization_agencies))
print("Matching agencies:", len(matching_agencies))


# =========================================================
# 7. Merge agency ratios with agency-ZIP demand
# =========================================================

df = demand.merge(
    agency_ratios,
    on="PRVDR_ID",
    how="left"
)

print("\nAfter merging:")
print("Rows:", len(df))
print("Agencies:", df["PRVDR_ID"].nunique())
print("ZIPs:", df["zip"].nunique())


# =========================================================
# 8. Identify missing agency ratios
# =========================================================

missing_ratio = (
    df["nursing_visits_per_episode"]
    .isna()
)

print("\nRows missing agency-specific nursing ratio:")
print(missing_ratio.sum())


# =========================================================
# 9. Fill missing ratios with overall median
# =========================================================

df["nursing_visits_per_episode_used"] = (
    df["nursing_visits_per_episode"]
    .fillna(fallback_ratio)
)


# =========================================================
# 10. Estimate nursing visits
# =========================================================

df["estimated_nursing_visits"] = (
    df["estimated_agency_zip_episodes"]
    * df["nursing_visits_per_episode_used"]
)


# =========================================================
# 11. Check for missing values
# =========================================================

print("\nMissing values:")

print(
    df[
        [
            "estimated_agency_zip_episodes",
            "nursing_visits_per_episode_used",
            "estimated_nursing_visits"
        ]
    ].isna().sum()
)


# =========================================================
# 12. Check for negative values
# =========================================================

print("\nNegative values:")

print(
    "Negative estimated episodes:",
    (
        df["estimated_agency_zip_episodes"] < 0
    ).sum()
)

print(
    "Negative nursing ratios:",
    (
        df["nursing_visits_per_episode_used"] < 0
    ).sum()
)

print(
    "Negative estimated visits:",
    (
        df["estimated_nursing_visits"] < 0
    ).sum()
)


# =========================================================
# 13. Display example results
# =========================================================

print("\nExample agency-ZIP nursing visit estimates:")

print(
    df[
        [
            "PRVDR_ID",
            "zip",
            "agency_total_episodes",
            "estimated_agency_zip_episodes",
            "nursing_visits_per_episode",
            "nursing_visits_per_episode_used",
            "estimated_nursing_visits"
        ]
    ].head(20)
)


# =========================================================
# 14. Summary statistics
# =========================================================

print("\nEstimated nursing visits summary:")

print(
    df["estimated_nursing_visits"]
    .describe()
)


# =========================================================
# 15. Calculate total estimated nursing visits
# =========================================================

total_estimated_visits = (
    df["estimated_nursing_visits"]
    .sum()
)

print("\nTotal estimated nursing visits:")
print(f"{total_estimated_visits:,.2f}")


# =========================================================
# 16. Save output
# =========================================================

df.to_csv(
    output_file,
    index=False
)

print("\nSaved output to:")
print(output_file)

print("\nFinal dataset shape:")
print(df.shape)