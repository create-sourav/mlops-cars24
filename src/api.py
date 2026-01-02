from fastapi import FastAPI
import joblib
import pandas as pd
import os

app = FastAPI(title="Cars24 Price Prediction API")

MODEL_PATH = "models/car_price_model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model file not found — train the model first or check models/ folder."
        )
    return joblib.load(MODEL_PATH)

model = load_model()

def preprocess(df: pd.DataFrame):
    # Extract Brand + Model (same as training)
    if "Car Name" in df.columns:
        df["Brand"] = df["Car Name"].str.split().str[0]
        df["Model"] = df["Car Name"].str.split().str[1:3].str.join(" ")
        df = df.drop(columns=["Car Name"])

    # One-hot encode (same logic as training)
    df = pd.get_dummies(df, drop_first=True, dtype=int)

    return df


@app.get("/")
def home():
    return {"message": "Cars24 Model API is running 🚗"}


@app.post("/predict")
def predict(data: list):
    """
    Accepts a list of dictionaries (JSON),
    converts to DataFrame, preprocesses, predicts,
    returns list with prediction values added.
    """

    df = pd.DataFrame(data)

    X = preprocess(df)
    preds = model.predict(X)

    df["Predicted_Price"] = preds.tolist()

    return df.to_dict(orient="records")
