import pandas as pd

# Load raw S1810 data
df = pd.read_csv("Datasets/Raw/S1810_2024.csv")

df = df[df["GEO_ID"].astype(str).str.startswith("1400000US")]

# Convert relevant Census columns to numeric
numeric_columns = [
    "S1810_C03_001E",
    "S1810_C03_019E",
    "S1810_C03_029E",
    "S1810_C03_039E",
    "S1810_C03_047E",
    "S1810_C03_055E",
    "S1810_C03_063E"
]

df[numeric_columns] = df[numeric_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

# Clean GEOID
df["GEOID"] = df["GEO_ID"].str.replace(
    "1400000US",
    "",
    regex=False
)

# Create processed disability dataset
disability_features = df[
    [
        "GEOID",
        "NAME",
        "S1810_C03_001E",
        "S1810_C03_019E",
        "S1810_C03_029E",
        "S1810_C03_039E",
        "S1810_C03_047E",
        "S1810_C03_055E",
        "S1810_C03_063E"
    ]
].copy()

# Rename columns
disability_features = disability_features.rename(
    columns={
        "S1810_C03_001E": "pct_disability",
        "S1810_C03_019E": "pct_hearing_diff",
        "S1810_C03_029E": "pct_vision_diff",
        "S1810_C03_039E": "pct_cognitive_diff",
        "S1810_C03_047E": "pct_ambulatory_diff",
        "S1810_C03_055E": "pct_selfcare_diff",
        "S1810_C03_063E": "pct_independent_living_diff"
    }
)

# View result
print("\nFinal disability dataframe:")
print(disability_features.head(10))

print("\nShape:")
print(disability_features.shape)

print("\nColumns:")
print(disability_features.columns.tolist())

# Save processed data
disability_features.to_csv(
    "Datasets/Processed/acs_disability_tract.csv",
    index=False
)