import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

MODEL_PATH = "models/car_price_model.pkl"
DATA_PATH = "data/processed/cars24_clean.csv"

def load_model():
    return joblib.load(MODEL_PATH)

def load_data():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("Price", axis=1)
    y = df["Price"]

    return X, y

def evaluate_model(model, X, y):
    preds = model.predict(X)

    mae = mean_absolute_error(y, preds)
    rmse = mean_squared_error(y, preds) ** 0.5 
    r2 = r2_score(y, preds)

    print("\n--- Evaluation ---")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.3f}")

if __name__ == "__main__":
    model = load_model()
    X, y = load_data()
    evaluate_model(model, X, y)
