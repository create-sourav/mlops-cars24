import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib
import os

# -------- PATHS --------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "cars24_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "car_price_model.pkl")



def load_data():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("Price", axis=1)
    y = df["Price"]

    # One-hot encode categorical columns
    X = pd.get_dummies(X, drop_first=True, dtype=int)

    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_model(x_train, y_train):

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBRegressor(
            n_estimators=200,
            max_depth=6,
            objective="reg:squarederror",
            random_state=42
        ))
    ])

    pipe.fit(x_train, y_train)
    return pipe


def save_model(model):
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print("Model saved at:", MODEL_PATH)


if __name__ == "__main__":
    x_train, x_test, y_train, y_test = load_data()
    model = train_model(x_train, y_train)
    save_model(model)
