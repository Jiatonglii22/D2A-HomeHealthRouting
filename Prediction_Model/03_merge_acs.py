import pandas as pd

# Load processed datasets
age = pd.read_csv("Datasets/Processed/acs_age_tract.csv")
disability = pd.read_csv("Datasets/Processed/acs_disability_tract.csv")

# Merge using Census tract GEOID
acs_tract = pd.merge(
    age,
    disability,
    on="GEOID",
    how="inner",
    suffixes=("", "_disability")
)

# Remove duplicate NAME column from disability dataset
acs_tract = acs_tract.drop(columns=["NAME_disability"])

# Remove zero-population tracts
acs_tract = acs_tract[
    acs_tract["population"] > 0
].copy()

print("Remaining tracts:", len(acs_tract))

print("\nRemaining missing values:")
print(acs_tract.isna().sum())

# View result
print("\nMerged ACS dataframe:")
print(acs_tract.head(10))

print("\nShape:")
print(acs_tract.shape)

print("\nColumns:")
print(acs_tract.columns.tolist())

# Check for missing values
print("\nMissing values:")
print(acs_tract.isna().sum())

# Save merged dataset
acs_tract.to_csv(
    "Datasets/Processed/acs_tract_features.csv",
    index=False
)

# print(
#     acs_tract[
#         acs_tract["pct_disability"].isna()
#     ][
#         ["GEOID", "NAME"]
#     ].to_string(index=False)
# )
# print("Population missing:")
# print(
#     acs_tract[acs_tract["population"].isna()][["GEOID", "NAME"]]
# )

# print("\nDisability missing:")
# print(
#     acs_tract[acs_tract["pct_disability"].isna()][["GEOID", "NAME"]]
# )

# missing_disability = acs_tract[
#     acs_tract["pct_disability"].isna()
# ][
#     [
#         "GEOID",
#         "NAME",
#         "population",
#         "pop_65_plus",
#         "pct_65_plus",
#         "pct_disability"
#     ]
# ]

# print(missing_disability.to_string(index=False))

# print(
#     acs_tract[
#         acs_tract["GEOID"].isin(missing_disability["GEOID"])
#     ][
#         ["GEOID", "population"]
#     ].to_string(index=False)
# )
