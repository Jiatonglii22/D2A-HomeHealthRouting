import pandas as pd

# 1. Load data
acs = pd.read_csv("Datasets/Processed/acs_tract_features.csv")
crosswalk = pd.read_excel("Datasets/Raw/ZIP-TRACT_122024.xlsx")

# 2. Clean IDs
acs["GEOID"] = acs["GEOID"].astype(str).str.zfill(11)
crosswalk["geoid"] = crosswalk["geoid"].astype(str).str.zfill(11)
crosswalk["zip"] = crosswalk["zip"].astype(str).str.zfill(5)

# 3. Keep Allegheny County
# Allegheny County FIPS = 42003
acs = acs[
    acs["GEOID"].str.startswith("42003")
].copy()

crosswalk = crosswalk[
    crosswalk["geoid"].str.startswith("42003")
].copy()

print("ACS tracts:", len(acs))
print("Crosswalk rows:", len(crosswalk))

# 4. Merge ACS with crosswalk
merged = crosswalk.merge(
    acs,
    left_on="geoid",
    right_on="GEOID",
    how="inner"
)

print("\nMerged rows:", len(merged))

# 5. Allocate tract counts to ZIP codes
merged["allocated_population"] = (
    merged["population"] * merged["res_ratio"]
)

merged["allocated_pop_65_plus"] = (
    merged["pop_65_plus"] * merged["res_ratio"]
)

merged["allocated_pop_75_plus"] = (
    merged["pop_75_plus"] * merged["res_ratio"]
)

merged["allocated_pop_85_plus"] = (
    merged["pop_85_plus"] * merged["res_ratio"]
)

# 6. Aggregate counts by ZIP
zip_features = (
    merged
    .groupby("zip")
    .agg(
        population=("allocated_population", "sum"),
        pop_65_plus=("allocated_pop_65_plus", "sum"),
        pop_75_plus=("allocated_pop_75_plus", "sum"),
        pop_85_plus=("allocated_pop_85_plus", "sum")
    )
    .reset_index()
)

# 7. Calculate age percentages
zip_features["pct_65_plus"] = (
    zip_features["pop_65_plus"]
    / zip_features["population"]
    * 100
)

zip_features["pct_75_plus"] = (
    zip_features["pop_75_plus"]
    / zip_features["population"]
    * 100
)

zip_features["pct_85_plus"] = (
    zip_features["pop_85_plus"]
    / zip_features["population"]
    * 100
)

# Remove ZIPs with no allocated residential population
zip_features = zip_features[
    zip_features["population"] > 0
].copy()

# 8. Population-weighted disability percentages
def weighted_average(group, value_column):

    weights = group["allocated_population"]

    valid = (
        group[value_column].notna()
        & weights.notna()
        & (weights > 0)
    )

    if valid.sum() == 0:
        return float("nan")

    return (
        (
            group.loc[valid, value_column]
            * group.loc[valid, "allocated_population"]
        ).sum()
        /
        group.loc[valid, "allocated_population"].sum()
    )


disability_columns = [
    "pct_disability",
    "pct_hearing_diff",
    "pct_vision_diff",
    "pct_cognitive_diff",
    "pct_ambulatory_diff",
    "pct_selfcare_diff",
    "pct_independent_living_diff"
]

for column in disability_columns:

    weighted = (
        merged
        .groupby("zip")
        .apply(
            lambda g: weighted_average(g, column),
            include_groups=False
        )
        .rename(column)
    )

    zip_features = zip_features.merge(
        weighted,
        on="zip",
        how="left"
    )

# 9. Clean numeric precision
count_columns = [
    "population",
    "pop_65_plus",
    "pop_75_plus",
    "pop_85_plus"
]

zip_features[count_columns] = (
    zip_features[count_columns].round(2)
)

percentage_columns = [
    "pct_65_plus",
    "pct_75_plus",
    "pct_85_plus",
    "pct_disability",
    "pct_hearing_diff",
    "pct_vision_diff",
    "pct_cognitive_diff",
    "pct_ambulatory_diff",
    "pct_selfcare_diff",
    "pct_independent_living_diff"
]

zip_features[percentage_columns] = (
    zip_features[percentage_columns].round(2)
)

# 10. Save
output_file = ("Datasets/Processed/acs_zip_features.csv")

zip_features.to_csv(output_file,index=False)

print("\nSaved:", output_file)

print("\nZIP-level dataset:")
print(
    zip_features.head(10).to_string(index=False)
)

print("\nShape:")
print(zip_features.shape)

print("\nMissing values:")
print(zip_features.isna().sum())

# Find ZIPs with missing age percentages
missing_zip = zip_features[
    zip_features["pct_65_plus"].isna()
]

print(
    missing_zip[
        [
            "zip",
            "population",
            "pop_65_plus",
            "pop_75_plus",
            "pop_85_plus"
        ]
    ].to_string(index=False)
)