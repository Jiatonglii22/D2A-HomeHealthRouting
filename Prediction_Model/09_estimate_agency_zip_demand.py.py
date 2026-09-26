import pandas as pd
import numpy as np


# =========================================================
# 1. Load agency-ZIP data
# =========================================================

input_file = "Datasets/Processed/agency_zip_features.csv"
output_file = "Datasets/Processed/agency_zip_demand.csv"

df = pd.read_csv(input_file)

print("Initial dataset shape:", df.shape)


# =========================================================
# 2. Keep agencies with available 2024 utilization
# =========================================================

# Do not treat missing CMS utilization as zero.
# Remove agencies that do not have 2024 utilization data.

df = df.dropna(subset=["agency_total_episodes"]).copy()

print("\nAfter removing agencies without 2024 utilization:")
print("Rows:", len(df))
print("Agencies:", df["PRVDR_ID"].nunique())
print("ZIPs:", df["zip"].nunique())


# =========================================================
# 3. Calculate ZIP-level need intensity
# =========================================================

# Need intensity:
#   60% = percentage of population age 65+
#   40% = percentage of population with a disability

df["need_rate"] = (
    0.60 * df["pct_65_plus"]
    + 0.40 * df["pct_disability"]
)


# =========================================================
# 4. Calculate raw ZIP-level demand score
# =========================================================

# Larger ZIPs have more potential demand.
#
# Demand score = population × need rate

df["raw_demand_score"] = (
    df["population"] * df["need_rate"]
)


# =========================================================
# 5. Calculate total demand score for each agency
# =========================================================

# Each agency distributes its total episodes across
# the ZIP codes that it serves.

df["agency_total_demand_score"] = (
    df.groupby("PRVDR_ID")["raw_demand_score"]
      .transform("sum")
)


# =========================================================
# 6. Allocate agency episodes across ZIP codes
# =========================================================

df["estimated_agency_zip_episodes"] = (
    df["agency_total_episodes"]
    * df["raw_demand_score"]
    / df["agency_total_demand_score"]
)


# =========================================================
# 7. Verify allocation
# =========================================================

allocation_check = (
    df.groupby("PRVDR_ID")["estimated_agency_zip_episodes"]
      .sum()
      .reset_index()
)

allocation_check = allocation_check.merge(
    df[["PRVDR_ID", "agency_total_episodes"]]
    .drop_duplicates(),
    on="PRVDR_ID"
)

allocation_check["difference"] = (
    allocation_check["estimated_agency_zip_episodes"]
    - allocation_check["agency_total_episodes"]
)

print("\nMaximum allocation difference:")
print(allocation_check["difference"].abs().max())

print("\nAllocation check:")
print(allocation_check.head(10))


# =========================================================
# 8. Check for missing values
# =========================================================

print("\nMissing values:")
print(df.isna().sum())


# =========================================================
# 9. Display example results
# =========================================================

print("\nExample agency-ZIP demand estimates:")

print(
    df[
        [
            "PRVDR_ID",
            "zip",
            "agency_total_episodes",
            "population",
            "pct_65_plus",
            "pct_disability",
            "need_rate",
            "raw_demand_score",
            "estimated_agency_zip_episodes"
        ]
    ].head(20)
)


# =========================================================
# 10. Summary statistics
# =========================================================

print("\nEstimated agency-ZIP episodes summary:")

print(
    df["estimated_agency_zip_episodes"].describe()
)


# =========================================================
# 11. Save final dataset
# =========================================================

df.to_csv(output_file, index=False)

print("\nSaved output to:")
print(output_file)

print("\nFinal dataset shape:")
print(df.shape)

print("\nFinal columns:")
print(df.columns.tolist())