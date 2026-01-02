import pandas as pd
import joblib
import os

MODEL_PATH = "models/car_price_model.pkl"
NEW_DATA_DIR = "data/new_data"
OUTPUT_DIR = "data/predictions"

def load_model():
    return joblib.load(MODEL_PATH)

def get_latest():
    files = [f for f in os.listdir(NEW_DATA_DIR) if f.endswith(".csv")]
    files.sort()
    return os.path.join(NEW_DATA_DIR, files[-1])

def predict():
    model = load_model()
    path = get_latest()

    df = pd.read_csv(path)

    # Extract brand/model like training
    df["Brand"] = df["Car Name"].str.split().str[0]
    df["Model"] = df["Car Name"].str.split().str[1:3].str.join(" ")
    df = df.drop(columns=["Car Name"])

    preds = model.predict(df)

    df["Predicted_Price"] = preds
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    out = os.path.join(OUTPUT_DIR, "predicted_output.csv")
    df.to_csv(out, index=False)

    print("Saved:", out)

if __name__ == "__main__":
    predict()
