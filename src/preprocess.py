import pandas as pd
import os

# ----- PATHS -----
# Get project base directory automatically
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "cars24_raw.csv")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "cars24_clean.csv")


def load_data():
    return pd.read_csv(RAW_PATH)


def clean_data(df):

    # Drop unwanted index column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Extract Brand + Model
    df["Brand"] = df["Car Name"].str.split().str[0]
    df["Model"] = df["Car Name"].str.split().str[1:3].str.join(" ")

    # Drop raw car-name
    df = df.drop(columns=["Car Name"])

    # Remove impossible values
    df = df[df["Price"] > 0]
    df = df[df["Distance"] >= 0]

    # ---- NEW: DROP MISSING VALUES ----
    df.dropna(inplace=True)

    # Handle rare models
    model_counts = df["Model"].value_counts()
    rare = model_counts[model_counts < 20].index
    df["Model"] = df["Model"].replace(rare, "Other")

    return df


def save_data(df):
    os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)


if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    save_data(df)
    print("Processed data saved:", PROCESSED_PATH)


print(df.shape)