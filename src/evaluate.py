import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cars24_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "car_price_model.pkl")


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = pd.get_dummies(df.drop("Price", axis=1), drop_first=True, dtype=int)
    y = df["Price"]
    return X, y

def load_model():
    return joblib.load(MODEL_PATH)

def evaluate_model(model, X, y):
    preds = model.predict(X)

    mae = mean_absolute_error(y, preds)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)

    print("\n--- Evaluation Results ---")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R²   : {r2:.3f}")

if __name__ == "__main__":
    X, y = load_data()
    model = load_model()
    evaluate_model(model, X, y)
