# 🚗 Cars24 Price Prediction — End-to-End MLOps Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-orange)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready machine learning system that predicts used car prices using an end-to-end pipeline with automated training, batch predictions, and real-time API inference.

This project demonstrates how companies like **Cars24**, **OLX**, and **CarDekho** deploy real-world ML systems in production.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Problem Statement](#-problem-statement)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [CI/CD Pipeline](#-cicd-pipeline)
- [API Documentation](#-api-documentation)
- [Model Evaluation](#-model-evaluation)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project showcases a complete MLOps workflow including:

- **Machine Learning**: XGBoost + Scikit-Learn Pipelines
- **CI/CD**: Automated training and testing with GitHub Actions
- **Batch Predictions**: Automated predictions on new data uploads
- **Real-Time API**: FastAPI service for instant predictions
- **Best Practices**: Pipeline preprocessing, model versioning, and separation of concerns

---

## ✨ Features

- ✅ Clean, maintainable ML pipeline with proper preprocessing
- ✅ Automated training and testing via CI/CD
- ✅ Model versioning and artifact management
- ✅ Automatic batch predictions using GitHub Actions
- ✅ Real-time model inference via REST API
- ✅ Integrated preprocessing (eliminates feature mismatch issues)
- ✅ Comprehensive evaluation metrics

---

## 🧠 Problem Statement

**Goal**: Predict the fair market selling price of used cars based on their attributes.

**Input Features**:
- Year of manufacture
- Distance driven (kilometers)
- Number of previous owners
- Fuel type (Petrol, Diesel, CNG, etc.)
- Drive type (Manual, Automatic)
- Car brand and model
- Location (registration code)
- Vehicle type (SUV, Sedan, Hatchback, etc.)

**Output**: Predicted selling price in INR

---

## 🏗️ Architecture

```
mlops_cars24/
│
├── data/
│   ├── raw/              # Original dataset
│   ├── processed/        # Cleaned dataset
│   ├── new_data/         # Incoming batch files for predictions
│   └── predictions/      # Output prediction files
│
├── models/
│   └── car_price_model.pkl   # Trained model pipeline
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py     # Data cleaning and preparation
│   ├── train.py          # Model training pipeline
│   ├── predict.py        # Batch prediction script
│   ├── api.py            # FastAPI application
│   └── evaluate.py       # Model evaluation metrics
│
├── .github/
│   └── workflows/
│       ├── ci.yml        # CI/CD: Train and test on push
│       └── predict.yml   # Auto-predict on new data uploads
│
├── tests/
│   └── test_model.py     # Unit tests
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/mlops_cars24.git
   cd mlops_cars24
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Usage

### 1️⃣ Data Preprocessing

Clean and prepare the raw dataset:

```bash
python src/preprocess.py
```

This will:
- Load data from `data/raw/`
- Handle missing values
- Extract features (brand, model, etc.)
- Save processed data to `data/processed/`

### 2️⃣ Model Training

Train the XGBoost model with the complete pipeline:

```bash
python src/train.py
```

The trained model pipeline will be saved to `models/car_price_model.pkl`

**Pipeline Components**:
- Numeric feature scaling
- Categorical feature encoding
- XGBoost regressor
- Automatic feature transformation

### 3️⃣ Batch Predictions

Run predictions on new data:

```bash
python src/predict.py --input data/new_data/batch_input.csv --output data/predictions/output.csv
```

### 4️⃣ Model Evaluation

Evaluate model performance:

```bash
python src/evaluate.py
```

**Metrics displayed**:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 🔄 CI/CD Pipeline

### Automated Training (`.github/workflows/ci.yml`)

Triggered on every push to the repository:

1. ✅ Install dependencies
2. ✅ Run preprocessing
3. ✅ Train the model
4. ✅ Execute tests
5. ✅ Validate pipeline integrity

### Automated Batch Predictions (`.github/workflows/predict.yml`)

Triggered when new CSV files are added to `data/new_data/`:

1. ✅ Load new data
2. ✅ Run predictions using trained model
3. ✅ Save results to `data/predictions/`
4. ✅ Upload predictions as GitHub Actions artifacts

**To use**:
1. Add your CSV file to `data/new_data/`
2. Commit and push
3. Download predictions from **Actions → Predict New Data → Artifacts**

---

## ⚡ API Documentation

### Starting the API Server

Run locally:

```bash
uvicorn src.api:app --reload
```

The API will be available at: `http://127.0.0.1:8000`

### Interactive Documentation

Visit Swagger UI: `http://127.0.0.1:8000/docs`

### Example API Request

**Endpoint**: `POST /predict`

**Request Body**:
```json
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
```

**Response**:
```json
{
  "Predicted_Price": 865432.21
}
```

### Using cURL

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Year": 2020,
    "Distance": 35000,
    "Owner": 1,
    "Fuel": "PETROL",
    "Location": "KA-05",
    "Drive": "Manual",
    "Type": "SUV",
    "Brand": "Hyundai",
    "Model": "Creta"
  }'
```

---

## 📊 Model Evaluation

The evaluation script provides comprehensive metrics:

```bash
python src/evaluate.py
```

**Sample Output**:
```
Model Evaluation Metrics:
========================
MAE:  45,234.56
RMSE: 67,890.12
R²:   0.8542
```

---

## 🛣️ Roadmap

Future enhancements planned for this project:

- [ ] Deploy API to cloud (Railway / Render / Hugging Face Spaces)
- [ ] Add API authentication and rate limiting
- [ ] Integrate MLflow for experiment tracking
- [ ] Implement scheduled retraining pipelines
- [ ] Build web UI dashboard for predictions
- [ ] Add monitoring and alerting
- [ ] Docker containerization
- [ ] Kubernetes deployment configuration

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please ensure:
- Code follows PEP 8 style guidelines
- All tests pass
- New features include appropriate tests

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: Cars24 scraped dataset (transformed for educational purposes)
- **Purpose**: Educational and portfolio demonstration
- **Inspiration**: Real-world MLOps practices from leading automotive platforms

---

## 📧 Contact

**Project Maintainer**: Your Name

- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your Profile](https://linkedin.com/in/yourprofile)
- Email: your.email@example.com

---

## 🎯 Key Learnings

This project demonstrates:

- ✅ Importance of integrated preprocessing in ML pipelines
- ✅ Preventing feature drift between training and inference
- ✅ Automating ML workflows with CI/CD
- ✅ Proper separation of batch, training, and API layers
- ✅ Building maintainable production ML systems (not just notebooks)

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star! ⭐**

</div>