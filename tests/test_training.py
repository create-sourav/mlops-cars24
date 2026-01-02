import sys, os

# 👇 Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
from src.train import load_data, train_model, save_model, MODEL_PATH


def test_training_pipeline():

    # 1️⃣ Data should load without errors
    x_train, x_test, y_train, y_test = load_data()
    assert len(x_train) > 0
    assert len(y_train) > 0

    # 2️⃣ Model should train
    model = train_model(x_train, y_train)
    assert model is not None

    # 3️⃣ Model should save
    save_model(model)
    assert os.path.exists(MODEL_PATH)

    # 4️⃣ Model should be loadable
    loaded = joblib.load(MODEL_PATH)
    assert loaded is not None


# run python -m pytest -q in terminal to execute the test now after this 
