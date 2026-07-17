"""
prep_adult_dataset.py

One-off script: adds column headers to the raw UCI Adult/Census Income
dataset (which ships headerless) and saves it as a proper CSV ready for
the real_world_stress_test.py script.

Run once with:
    uv run python prep_adult_dataset.py
"""

import pandas as pd

COLUMN_NAMES = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income",
]

RAW_PATH = "data/external/adult_raw.csv"
OUT_PATH = "data/external/adult_census_income.csv"

df = pd.read_csv(RAW_PATH, header=None, names=COLUMN_NAMES, skipinitialspace=True)

# UCI encodes missing values as the literal string "?" -- leave this AS-IS
# on purpose. Converting it to real NaN here would defeat the point of the
# stress test, which is to see whether the PIPELINE ITSELF (via profiler.py's
# text-column detection) handles this real-world missing-value convention,
# not whether we can pre-clean it away before the agent ever sees it.

# income is the target column: values are " <=50K" / " >50K" (or with a
# trailing "." in some UCI mirrors) -- normalize to a clean binary label.
df["income"] = df["income"].str.strip().str.rstrip(".")
df["target"] = (df["income"] == ">50K").astype(int)
df = df.drop(columns=["income"])

df.to_csv(OUT_PATH, index=False)
print(f"Saved {len(df)} rows, {len(df.columns)} columns to {OUT_PATH}")
print(f"Target balance:\n{df['target'].value_counts()}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nSample of '?' missing-value convention (workclass column):")
print(df['workclass'].value_counts().head())
