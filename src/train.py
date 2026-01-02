import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

DATA_PATH = "data/processed/cars24_clean.csv"
MODEL_PATH = "models/car_price_model.pkl"


def load_data(path: str = DATA_PATH):
    print(f"\n📥 Loading data: {path}")
    df = pd.read_csv(path)

    X = df.drop("Price", axis=1)
    y = df["Price"]

    return train_test_split(X, y, test_size=0.2, random_state=42)


def build_pipeline():
    numeric = ["Year", "Distance", "Owner"]
    categorical = ["Fuel", "Location", "Drive", "Type", "Brand", "Model"]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        objective="reg:squarederror",
        random_state=42,
    )

    return Pipeline(steps=[("preprocess", preprocess), ("model", model)])


def train_model(X_train, y_train):
    print("🚀 Training model...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    return pipeline


def save_model(model, path: str = MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"✔ MODEL SAVED: {os.path.abspath(path)}")


def main():
    X_train, X_test, Y_train, Y_test = load_data()
    model = train_model(X_train, Y_train)
    save_model(model)

    print("\n🔎 MODEL TYPE:", type(model))
    print("🎯 TRAINED ON COLUMNS:", model.named_steps["preprocess"].get_feature_names_out())
    print("\n👍 Training finished.\n")


if __name__ == "__main__":
    main()
