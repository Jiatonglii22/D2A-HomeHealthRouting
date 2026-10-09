import pandas as pd

# Load ACS B01001 data 
df = pd.read_csv("Datasets/Raw/B01001_2024.csv")

print(df.head())

df = df[df["GEO_ID"].astype(str).str.startswith("1400000US")]

# Convert Census estimate columns from strings to numbers
numeric_columns = [
    "B01001_001E",
    "B01001_020E",
    "B01001_021E",
    "B01001_022E",
    "B01001_023E",
    "B01001_024E",
    "B01001_025E",
    "B01001_044E",
    "B01001_045E",
    "B01001_046E",
    "B01001_047E",
    "B01001_048E",
    "B01001_049E"
]

df[numeric_columns] = df[numeric_columns].apply(
    pd.to_numeric,
    errors="coerce"
)

# Total Population
df["population"] = df["B01001_001E"]

# Population 65+
df["pop_65_plus"] = (
    df["B01001_020E"] +
    df["B01001_021E"] +
    df["B01001_022E"] +
    df["B01001_023E"] +
    df["B01001_024E"] +
    df["B01001_025E"] +
    df["B01001_044E"] +
    df["B01001_045E"] +
    df["B01001_046E"] +
    df["B01001_047E"] +
    df["B01001_048E"] +
    df["B01001_049E"]
)

df["pct_65_plus"] = (
    df["pop_65_plus"] /
    df["population"] * 100
)

# Population 75+
df["pop_75_plus"] = (
    df["B01001_023E"] +
    df["B01001_024E"] +
    df["B01001_025E"] +
    df["B01001_047E"] +
    df["B01001_048E"] +
    df["B01001_049E"]
)

df["pct_75_plus"] = (
    df["pop_75_plus"] /
    df["population"] * 100
)

# Population 85+
df["pop_85_plus"] = (
    df["B01001_025E"] +
    df["B01001_049E"]
)

df["pct_85_plus"] = (
    df["pop_85_plus"] /
    df["population"] * 100
)

# Clean GEOID
df["GEOID"] = df["GEO_ID"].str.replace(
    "1400000US",
    "",
    regex=False
)

df["GEOID"] = df["GEOID"].astype(str)

# Select final variables
tract_features = df[
    [
        "GEOID",
        "NAME",
        "population",
        "pop_65_plus",
        "pct_65_plus",
        "pop_75_plus",
        "pct_75_plus",
        "pop_85_plus",
        "pct_85_plus"
    ]
]

print("\nFinal dataframe columns:")
print(tract_features.columns.tolist())

print("\nFirst 10 rows:")
print(tract_features.head(10))

print("\nDataframe shape:")
print(tract_features.shape)

# Save processed data
tract_features.to_csv(
    "Datasets/Processed/acs_age_tract.csv",
    index=False
)