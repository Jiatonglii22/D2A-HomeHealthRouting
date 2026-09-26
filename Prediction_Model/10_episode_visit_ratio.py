import pandas as pd


# =========================================================
# 1. Load CMS home health utilization data
# =========================================================

input_file = "Datasets/Raw/HH_UtilizationZip_2024.csv"

df = pd.read_csv(input_file)

print("Dataset shape:", df.shape)


# =========================================================
# 2. Keep the 2024 Home Health records
# =========================================================

df = df[
    (df["YEAR"] == 2024) &
    (df["YEAR_TYPE"] == "CY") &
    (df["SRVC_CTGRY"] == "HH")
].copy()

print("\nFiltered dataset shape:", df.shape)


# =========================================================
# 3. Convert utilization columns to numeric
# =========================================================

numeric_columns = [
    "TOT_EPSD_STAY_CNT",
    "TOT_SRVC_DAYS",
    "NRSNG_VISITS_CNT",
    "MSW_VISITS_CNT",
    "AIDE_VISITS_CNT"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )


# =========================================================
# 4. Examine the relevant utilization variables
# =========================================================

columns_to_check = [
    "TOT_EPSD_STAY_CNT",
    "TOT_SRVC_DAYS",
    "NRSNG_VISITS_CNT",
    "MSW_VISITS_CNT",
    "AIDE_VISITS_CNT"
]

print("\nRelevant CMS utilization columns:")

print(
    df[columns_to_check].describe()
)


# =========================================================
# 5. Calculate total episodes and visits
# =========================================================

total_episodes = df["TOT_EPSD_STAY_CNT"].sum()

total_service_days = df["TOT_SRVC_DAYS"].sum()

total_nursing_visits = df["NRSNG_VISITS_CNT"].sum()

total_msw_visits = df["MSW_VISITS_CNT"].sum()

total_aide_visits = df["AIDE_VISITS_CNT"].sum()


# =========================================================
# 6. Calculate visits per episode
# =========================================================

service_days_per_episode = (
    total_service_days / total_episodes
)

nursing_visits_per_episode = (
    total_nursing_visits / total_episodes
)

msw_visits_per_episode = (
    total_msw_visits / total_episodes
)

aide_visits_per_episode = (
    total_aide_visits / total_episodes
)


# =========================================================
# 7. Print overall results
# =========================================================

print("\n==============================")
print("2024 CMS HOME HEALTH SUMMARY")
print("==============================")

print(
    f"Total episodes: "
    f"{total_episodes:,.0f}"
)

print(
    f"Total service days: "
    f"{total_service_days:,.0f}"
)

print(
    f"Total nursing visits: "
    f"{total_nursing_visits:,.0f}"
)

print(
    f"Total MSW visits: "
    f"{total_msw_visits:,.0f}"
)

print(
    f"Total aide visits: "
    f"{total_aide_visits:,.0f}"
)


print("\n==============================")
print("UTILIZATION PER EPISODE")
print("==============================")

print(
    f"Service days per episode: "
    f"{service_days_per_episode:.2f}"
)

print(
    f"Nursing visits per episode: "
    f"{nursing_visits_per_episode:.2f}"
)

print(
    f"MSW visits per episode: "
    f"{msw_visits_per_episode:.2f}"
)

print(
    f"Aide visits per episode: "
    f"{aide_visits_per_episode:.2f}"
)


# =========================================================
# 8. Calculate total visits
# =========================================================

total_visits = (
    total_nursing_visits
    + total_msw_visits
    + total_aide_visits
)

print("\n==============================")
print("TOTAL VISITS")
print("==============================")

print(
    f"Total visits: "
    f"{total_visits:,.0f}"
)

print(
    f"Total visits per episode: "
    f"{total_visits / total_episodes:.2f}"
)


# =========================================================
# 9. Calculate agency-level utilization
# =========================================================

agency_summary = (
    df.groupby(
        ["PRVDR_ID", "PRVDR_NAME"],
        as_index=False
    )
    .agg(
        agency_episodes=(
            "TOT_EPSD_STAY_CNT",
            "sum"
        ),

        agency_service_days=(
            "TOT_SRVC_DAYS",
            "sum"
        ),

        agency_nursing_visits=(
            "NRSNG_VISITS_CNT",
            "sum"
        ),

        agency_msw_visits=(
            "MSW_VISITS_CNT",
            "sum"
        ),

        agency_aide_visits=(
            "AIDE_VISITS_CNT",
            "sum"
        )
    )
)


# =========================================================
# 10. Calculate agency-level ratios
# =========================================================

agency_summary["service_days_per_episode"] = (
    agency_summary["agency_service_days"]
    / agency_summary["agency_episodes"]
)

agency_summary["nursing_visits_per_episode"] = (
    agency_summary["agency_nursing_visits"]
    / agency_summary["agency_episodes"]
)

agency_summary["msw_visits_per_episode"] = (
    agency_summary["agency_msw_visits"]
    / agency_summary["agency_episodes"]
)

agency_summary["aide_visits_per_episode"] = (
    agency_summary["agency_aide_visits"]
    / agency_summary["agency_episodes"]
)


# =========================================================
# 11. Calculate total visits per agency
# =========================================================

agency_summary["total_visits"] = (
    agency_summary["agency_nursing_visits"]
    + agency_summary["agency_msw_visits"]
    + agency_summary["agency_aide_visits"]
)

agency_summary["total_visits_per_episode"] = (
    agency_summary["total_visits"]
    / agency_summary["agency_episodes"]
)


# =========================================================
# 12. Display agency-level results
# =========================================================

print("\n==============================")
print("AGENCY-LEVEL UTILIZATION")
print("==============================")

print(
    agency_summary[
        [
            "PRVDR_ID",
            "PRVDR_NAME",
            "agency_episodes",
            "service_days_per_episode",
            "nursing_visits_per_episode",
            "msw_visits_per_episode",
            "aide_visits_per_episode",
            "total_visits_per_episode"
        ]
    ].head(20)
)


# =========================================================
# 13. Summary statistics for agencies
# =========================================================

print("\n==============================")
print("AGENCY RATIO SUMMARY")
print("==============================")

print(
    agency_summary[
        [
            "service_days_per_episode",
            "nursing_visits_per_episode",
            "msw_visits_per_episode",
            "aide_visits_per_episode",
            "total_visits_per_episode"
        ]
    ].describe()
)


# =========================================================
# 14. Save agency-level summary
# =========================================================

output_file = (
    "Datasets/Processed/"
    "cms_agency_utilization_summary.csv"
)

agency_summary.to_csv(
    output_file,
    index=False
)

print("\nSaved agency summary to:")
print(output_file)