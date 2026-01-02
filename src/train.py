import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
import joblib
import os

RAW_PATH = "data/processed/cars24_clean.csv"
MODEL_PATH = "models/car_price_model.pkl"

print("\n📥 Loading data:", RAW_PATH)
df = pd.read_csv(RAW_PATH)

# -----------------------------
# 1️⃣  ENSURE ONLY RAW FEATURES
# -----------------------------
features = [
    "Year", "Distance", "Owner",
    "Fuel", "Location", "Drive", "Type",
    "Brand", "Model"
]

target = "Price"

df = df[features + [target]]

X = df[features]
y = df[target]

numeric = ["Year", "Distance", "Owner"]
categorical = ["Fuel", "Location", "Drive", "Type", "Brand", "Model"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ]
)

model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    objective="reg:squarederror",
    random_state=42
)

pipeline = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", model),
    ]
)

print("🚀 Training model...")
pipeline.fit(X, y)

os.makedirs("models", exist_ok=True)

# ALWAYS overwrite
joblib.dump(pipeline, MODEL_PATH)

print("✔ MODEL SAVED:", os.path.abspath(MODEL_PATH))
print("\n🔎 MODEL TYPE:", type(pipeline))
print("🎯 TRAINED ON COLUMNS:", list(X.columns))
print("\n👍 Training finished.\n")
