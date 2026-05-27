import pandas as pd

# Load only small subset first
df = pd.read_csv("medium-english-50mb.csv", nrows=100)

print("Dataset loaded successfully")
print("Number of articles:", len(df))

print("\nColumns:")
print(df.columns)

print("\nFirst article preview:")
print(df.iloc[0]["title"])
print(df.iloc[0]["text"][:500])
