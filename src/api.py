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


@app.get("/")
def home():
    return {"message": "Cars24 Model API is running 🚗"}


@app.post("/predict")
def predict(data: dict):
    """
    Accepts ONE JSON car record
    Example:
    {
      "Year": 2020,
      "Distance": 35000,
      "Owner": 1,
      "Fuel": "PETROL",
      "Location": "KA-05",
      "Drive": "Manual",
      "Type": "SUV",
      "Brand": "Hyundai",
      "Model": "Creta"
    }
    """

    df = pd.DataFrame([data])

    preds = model.predict(df)

    return {"Predicted_Price": float(preds[0])}
