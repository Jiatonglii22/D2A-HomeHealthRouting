import pandas as pd

crosswalk = pd.read_excel(
    "Datasets/Raw/ZIP-TRACT_122024.xlsx"
)

print("Shape:")
print(crosswalk.shape)

print("\nColumns:")
print(crosswalk.columns.tolist())

print("\nFirst 10 rows:")
print(crosswalk.head(10).to_string(index=False))