import pandas as pd

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

utilization_file = "Datasets/Raw/HH_UtilizationZip_2024.csv"
service_zip_file = "Datasets/Raw/HH_ServicedZip_Jul2026.csv"
acs_zip_file = "Datasets/Processed/acs_zip_features.csv"

output_file = "Datasets/Processed/cms_zip_demand.csv"


# --------------------------------------------------
# 2. Load datasets
# --------------------------------------------------

utilization = pd.read_csv(utilization_file)
service_zips = pd.read_csv(service_zip_file)
acs_zips = pd.read_csv(acs_zip_file, dtype={"zip": str})


# --------------------------------------------------
# 3. Clean column names
# --------------------------------------------------

utilization.columns = utilization.columns.str.strip()
service_zips.columns = service_zips.columns.str.strip()


# --------------------------------------------------
# 4. Rename service ZIP columns
# --------------------------------------------------

service_zips = service_zips.rename(columns={
    "CMS Certification Number (CCN)": "PRVDR_ID",
    "ZIP Code": "zip"
})


# --------------------------------------------------
# 5. Clean ZIP codes and provider IDs
# --------------------------------------------------

service_zips["zip"] = (
    service_zips["zip"]
    .astype(str)
    .str.extract(r"(\d{5})")[0]
)

service_zips["PRVDR_ID"] = (
    service_zips["PRVDR_ID"]
    .astype(str)
    .str.strip()
)

utilization["PRVDR_ID"] = (
    utilization["PRVDR_ID"]
    .astype(str)
    .str.strip()
)


# --------------------------------------------------
# 6. Get Allegheny County ZIP codes
# --------------------------------------------------

allegheny_zips = set(
    acs_zips["zip"]
    .astype(str)
    .str.zfill(5)
)


# --------------------------------------------------
# 7. Keep only service relationships involving
#    Allegheny County ZIPs
# --------------------------------------------------

service_zips = service_zips[
    service_zips["zip"].isin(allegheny_zips)
].copy()


# --------------------------------------------------
# 8. Remove duplicate agency-ZIP relationships
# --------------------------------------------------

service_zips = service_zips[
    ["PRVDR_ID", "zip"]
].drop_duplicates()


# --------------------------------------------------
# 9. Check how many Allegheny ZIPs each agency serves
# --------------------------------------------------

agency_zip_counts = (
    service_zips
    .groupby("PRVDR_ID")["zip"]
    .nunique()
    .reset_index(name="num_allegheny_zips_served")
)


# --------------------------------------------------
# 10. Prepare utilization
# --------------------------------------------------

utilization["YEAR"] = pd.to_numeric(
    utilization["YEAR"],
    errors="coerce"
)

utilization["TOT_EPSD_STAY_CNT"] = pd.to_numeric(
    utilization["TOT_EPSD_STAY_CNT"],
    errors="coerce"
)

utilization = utilization[
    utilization["TOT_EPSD_STAY_CNT"].notna()
].copy()


# --------------------------------------------------
# 11. Attach number of Allegheny ZIPs served
# --------------------------------------------------

utilization = utilization.merge(
    agency_zip_counts,
    on="PRVDR_ID",
    how="inner"
)


# --------------------------------------------------
# 12. Equal allocation across Allegheny ZIPs
# --------------------------------------------------

utilization["estimated_episodes_per_zip"] = (
    utilization["TOT_EPSD_STAY_CNT"]
    / utilization["num_allegheny_zips_served"]
)


# --------------------------------------------------
# 13. Attach each agency to its Allegheny ZIPs
# --------------------------------------------------

agency_zip_demand = utilization[
    [
        "YEAR",
        "PRVDR_ID",
        "TOT_EPSD_STAY_CNT",
        "num_allegheny_zips_served",
        "estimated_episodes_per_zip"
    ]
].merge(
    service_zips,
    on="PRVDR_ID",
    how="inner"
)


# --------------------------------------------------
# 14. Aggregate agencies within each ZIP
# --------------------------------------------------

zip_demand = (
    agency_zip_demand
    .groupby(["zip", "YEAR"])
    .agg(
        estimated_home_health_episodes=(
            "estimated_episodes_per_zip",
            "sum"
        ),
        number_of_agencies_serving=(
            "PRVDR_ID",
            "nunique"
        )
    )
    .reset_index()
)


# --------------------------------------------------
# 15. Rename and clean
# --------------------------------------------------

zip_demand = zip_demand.rename(
    columns={"YEAR": "year"}
)

zip_demand["zip"] = (
    zip_demand["zip"]
    .astype(str)
    .str.zfill(5)
)

zip_demand = zip_demand.sort_values(
    ["zip", "year"]
).reset_index(drop=True)


# --------------------------------------------------
# 16. Save
# --------------------------------------------------

zip_demand.to_csv(
    output_file,
    index=False
)


# --------------------------------------------------
# 17. Print results
# --------------------------------------------------

print("\nFinal ZIP-level demand:")
print(zip_demand.head(20))

print("\nShape:")
print(zip_demand.shape)

print("\nNumber of ZIPs:")
print(zip_demand["zip"].nunique())

print("\nYears:")
print(sorted(zip_demand["year"].unique()))

print("\nMissing values:")
print(zip_demand.isna().sum())

print(f"\nSaved to: {output_file}")