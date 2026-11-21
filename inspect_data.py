from scipy.io import arff
import pandas as pd

data, _ = arff.loadarff("data/raw/jm1.arff")
df = pd.DataFrame(data)

print("Rows and columns:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nDefect labels:")
df["label"] = df["label"].str.decode("utf-8")
print(df["label"].value_counts())
print("\nAll column names:")
for column in df.columns:
    print(column)
