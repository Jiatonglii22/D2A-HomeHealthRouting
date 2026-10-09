import pandas as pd


# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

service_zip_file = "Datasets/Raw/HH_ServicedZip_Jul2026.csv"
utilization_file = "Datasets/Raw/HH_UtilizationZip_2024.csv"
acs_zip_file = "Datasets/Processed/acs_zip_features.csv"

output_file = "Datasets/Processed/agency_zip_features.csv"


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

service_zips = pd.read_csv(service_zip_file)

utilization = pd.read_csv(
    utilization_file
)

acs = pd.read_csv(
    acs_zip_file,
    dtype={"zip": str}
)


# --------------------------------------------------
# 3. Clean column names
# --------------------------------------------------

service_zips.columns = (
    service_zips.columns.str.strip()
)

utilization.columns = (
    utilization.columns.str.strip()
)


# --------------------------------------------------
# 4. Rename CMS service ZIP columns
# --------------------------------------------------

service_zips = service_zips.rename(
    columns={
        "CMS Certification Number (CCN)": "PRVDR_ID",
        "ZIP Code": "zip"
    }
)


# --------------------------------------------------
# 5. Clean agency IDs
# --------------------------------------------------

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
# 6. Clean ZIP codes
# --------------------------------------------------

service_zips["zip"] = (
    service_zips["zip"]
    .astype(str)
    .str.extract(r"(\d{5})")[0]
)

acs["zip"] = (
    acs["zip"]
    .astype(str)
    .str.zfill(5)
)


# --------------------------------------------------
# 7. Get Allegheny County ZIPs
# --------------------------------------------------

allegheny_zips = set(
    acs["zip"]
)


# Keep only service relationships involving
# Allegheny County ZIP codes

service_zips = service_zips[
    service_zips["zip"].isin(allegheny_zips)
].copy()


# Remove duplicate agency-ZIP relationships

service_zips = service_zips[
    ["PRVDR_ID", "zip"]
].drop_duplicates()


# --------------------------------------------------
# 8. Keep 2024 Home Health utilization
# --------------------------------------------------

utilization["YEAR"] = pd.to_numeric(
    utilization["YEAR"],
    errors="coerce"
)

utilization["NRSNG_VISITS_CNT"] = pd.to_numeric(
    utilization["NRSNG_VISITS_CNT"],
    errors="coerce"
)

utilization_2024 = utilization[
    (utilization["YEAR"] == 2024)
    &
    (utilization["YEAR_TYPE"] == "CY")
    &
    (utilization["SRVC_CTGRY"] == "HH")
    &
    (utilization["NRSNG_VISITS_CNT"].notna())
].copy()


# --------------------------------------------------
# 9. Calculate annual nursing visits per agency
# --------------------------------------------------

agency_utilization = (
    utilization_2024
    .groupby("PRVDR_ID", as_index=False)
    .agg(
        agency_total_nursing_visits=(
            "NRSNG_VISITS_CNT",
            "sum"
        )
    )
)


# --------------------------------------------------
# 10. Count agencies serving each ZIP
# --------------------------------------------------

zip_agency_counts = (
    service_zips
    .groupby("zip")["PRVDR_ID"]
    .nunique()
    .reset_index(
        name="number_of_agencies_serving_zip"
    )
)


# --------------------------------------------------
# 11. Add ZIP-level agency count
# --------------------------------------------------

agency_zip = service_zips.merge(
    zip_agency_counts,
    on="zip",
    how="left"
)


# --------------------------------------------------
# 12. Add annual agency nursing visits
# --------------------------------------------------

agency_zip = agency_zip.merge(
    agency_utilization,
    on="PRVDR_ID",
    how="left"
)


# --------------------------------------------------
# 13. Add ACS ZIP characteristics
# --------------------------------------------------

agency_zip = agency_zip.merge(
    acs,
    on="zip",
    how="left"
)


# --------------------------------------------------
# 14. Sort
# --------------------------------------------------

agency_zip = agency_zip.sort_values(
    ["PRVDR_ID", "zip"]
).reset_index(drop=True)


# --------------------------------------------------
# 15. Print diagnostics
# --------------------------------------------------

print("\nAgency × ZIP dataset:")
print(
    agency_zip.head(20)
)

print("\nShape:")
print(
    agency_zip.shape
)

print("\nNumber of agencies:")
print(
    agency_zip["PRVDR_ID"].nunique()
)

print("\nNumber of ZIPs:")
print(
    agency_zip["zip"].nunique()
)

print("\nMissing values:")
print(
    agency_zip.isna().sum()
)


# --------------------------------------------------
# 16. Display nursing visit summary
# --------------------------------------------------

print("\nAnnual nursing visits per agency:")

print(
    agency_zip[
        [
            "PRVDR_ID",
            "agency_total_nursing_visits"
        ]
    ]
    .drop_duplicates()
    .describe()
)


# --------------------------------------------------
# 17. Save
# --------------------------------------------------

agency_zip.to_csv(
    output_file,
    index=False
)

print(
    f"\nSaved to: {output_file}"
)


# --------------------------------------------------
# 18. Investigate agencies without
#     2024 nursing utilization
# --------------------------------------------------

missing_agencies = (
    agency_zip[
        agency_zip[
            "agency_total_nursing_visits"
        ].isna()
    ]["PRVDR_ID"]
    .drop_duplicates()
    .tolist()
)

print(
    "\nNumber of agencies without "
    "2024 nursing utilization:"
)

print(
    len(missing_agencies)
)

print(
    "\nAgencies without 2024 nursing utilization:"
)

print(
    missing_agencies
)

print(
    "\nNumber of agency-ZIP rows affected:"
)

print(
    agency_zip[
        "agency_total_nursing_visits"
    ].isna().sum()
)