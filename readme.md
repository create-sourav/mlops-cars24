# 🚗 Cars24 Price Prediction — End-to-End MLOps Project

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-orange)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Live API](https://img.shields.io/badge/Live%20API-Render-brightgreen)](https://mlops-cars24-2.onrender.com/docs#/default/home__get)

A production-ready machine learning system that predicts used car prices using an end-to-end pipeline with automated training, batch predictions, and real-time API inference.

This project demonstrates how companies like **Cars24**, **OLX**, and **CarDekho** deploy real-world ML systems in production.

🌐 **[Live API Documentation](https://mlops-cars24-2.onrender.com/docs#/default/home__get)**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Problem Statement](#-problem-statement)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage Guide](#-usage-guide)
  - [Data Preprocessing](#1️⃣-data-preprocessing)
  - [Model Training](#2️⃣-model-training)
  - [Retraining with New Data](#3️⃣-retraining-with-new-data)
  - [Batch Predictions](#4️⃣-batch-predictions)
  - [Model Evaluation](#5️⃣-model-evaluation)
- [CI/CD Pipeline](#-cicd-pipeline)
- [API Documentation](#-api-documentation)
- [FAQ](#-frequently-asked-questions)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This project showcases a complete MLOps workflow including:

- **Machine Learning**: XGBoost + Scikit-Learn Pipelines
- **CI/CD**: Automated training and testing with GitHub Actions
- **Batch Predictions**: Automated predictions on new data uploads
- **Real-Time API**: FastAPI service deployed on Render
- **Best Practices**: Pipeline preprocessing, model versioning, and separation of concerns

---

## ✨ Features

- ✅ Clean, maintainable ML pipeline with proper preprocessing
- ✅ Automated training and testing via CI/CD
- ✅ Model versioning and artifact management
- ✅ Automatic batch predictions using GitHub Actions
- ✅ Real-time model inference via REST API (deployed on Render)
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
│   ├── raw/                  # Original incoming dataset (never modified)
│   ├── processed/            # Clean/feature-engineered data
│   ├── new_data/             # New CSVs uploaded for predictions
│   └── predictions/          # Output prediction files
│
├── models/
│   └── car_price_model.pkl   # Trained ML pipeline (production)
│
├── notebooks/
│   └── model_selection.ipynb # EXPERIMENTS: model comparison & tuning notes
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py         # Data cleaning pipeline
│   ├── train.py              # Training script (saves new model)
│   ├── evaluate.py           # Metrics after training
│   ├── predict.py            # Batch prediction script
│   └── api.py                # FastAPI serving model predictions
│
├── tests/
│   └── test_model.py         # Unit tests for model pipeline
│
├── .github/
│   └── workflows/
│       ├── ci.yml            # CI/CD: test + train on push
│       └── predict.yml       # Auto-predict when new CSV uploaded
│
├── .pytest_cache/            # pytest cache (NOT committed to git)
├── .gitignore
├── requirements.txt
├── Procfile                  # Render/Heroku API deploy config
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

## 💻 Usage Guide

### 1️⃣ Data Preprocessing

Clean and prepare the raw dataset:

```bash
python src/preprocess.py
```

This will:
- Load data from `data/raw/cars24.csv`
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

### 3️⃣ Retraining with New Data

**To retrain the model with new data:**

1. **Add new training data** to `data/raw/cars24.csv`
   - Append new rows to the existing file
   - ⚠️ **Important**: Do NOT delete historical records. More data = better model.

2. **Commit and push**:
   ```bash
   git add data/raw/cars24.csv
   git commit -m "Add new training data"
   git push
   ```

3. **Automatic retraining happens via CI/CD**:
   - GitHub Actions pipeline runs automatically
   - `preprocess.py` creates updated processed dataset
   - `train.py` re-trains the model
   - New model file saved to `models/car_price_model.pkl`
   - CI uploads it as an artifact

4. **API model update**:
   - After retraining completes, push triggers redeployment
   - Render automatically deploys the app
   - API loads the newest model

✅ **Retraining workflow**: Add data → Push → Pipeline runs → Model updates

### 4️⃣ Batch Predictions

**Two ways to generate predictions:**

#### Method A: Local Prediction

Run predictions on new data locally:

```bash
python src/predict.py
```

This will:
- Read CSV from `data/new_data/`
- Use trained model to predict
- Save results to `data/predictions/predicted_output.csv`

#### Method B: Automated via GitHub Actions

1. Add your CSV file to `data/new_data/`
2. Commit and push:
   ```bash
   git add data/new_data/
   git commit -m "Add new data for prediction"
   git push
   ```
3. GitHub Actions automatically:
   - Reads latest file from `data/new_data/`
   - Runs predictions using trained model
   - Saves results to `data/predictions/predicted_output.csv`
   - Uploads predictions as artifacts

4. Download predictions: **Actions → Predict New Data → Artifacts**

### 5️⃣ Model Evaluation

Evaluate model performance:

```bash
python src/evaluate.py
```

**Metrics displayed**:
- Mean Absolute Error (MAE) : 40332.82
- Root Mean Squared Error (RMSE): 60742.43
- R² Score : 0.948

---

## 🔄 CI/CD Pipeline

### Automated Training (`.github/workflows/ci.yml`)

**Triggered on**: Every push to the repository

**Pipeline steps**:
1. ✅ Install dependencies
2. ✅ Run preprocessing
3. ✅ Train the model
4. ✅ Execute tests
5. ✅ Validate pipeline integrity
6. ✅ Upload model as artifact

### Automated Batch Predictions (`.github/workflows/predict.yml`)

**Triggered when**: New CSV files are added to `data/new_data/`

**Pipeline steps**:
1. ✅ Load new data from `data/new_data/`
2. ✅ Run predictions using trained model
3. ✅ Save results to `data/predictions/`
4. ✅ Upload predictions as GitHub Actions artifacts

---

## ⚡ API Documentation

### Live API

🌐 **Production API**: [https://mlops-cars24-2.onrender.com/docs](https://mlops-cars24-2.onrender.com/docs#/default/home__get)

### Running Locally

Start the API server:

```bash
uvicorn src.api:app --reload
```

Local access: `http://127.0.0.1:8000`

Interactive docs: `http://127.0.0.1:8000/docs`

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

### Using cURL (Production API)

```bash
curl -X POST "https://mlops-cars24-2.onrender.com/predict" \
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

## 🔍 Frequently Asked Questions

### Q1: How do I feed new data to retrain the model?

**Answer**: 
1. Add new rows to `data/raw/cars24.csv` (append, don't delete old data)
2. Commit and push:
   ```bash
   git add data/raw/cars24.csv
   git commit -m "Add new training data"
   git push
   ```
3. GitHub Actions automatically retrains the model
4. New model is saved and deployed

### Q2: How do I get predictions for new data?

**Answer**: You have two options:

| Method | Use Case | Output |
|--------|----------|--------|
| **Batch (CSV)** | Bulk predictions | CSV file in `data/predictions/` |
| **API** | Single/real-time predictions | JSON response instantly |

**Batch method**:
- Put CSV in `data/new_data/`
- Run `python src/predict.py` OR push to GitHub
- Get results in `data/predictions/predicted_output.csv`

**API method**:
- Send POST request to API endpoint
- Get instant JSON response

Both use the same trained model.

### Q3: Does retraining automatically update my API model?

**Answer**: Yes, with the current Render deployment setup:

1. You push new training data
2. GitHub Actions retrains the model
3. Push triggers Render auto-deployment
4. API loads the newest `car_price_model.pkl`

The API uses the latest model file in the repository.

### Q4: What is the `.pytest_cache/` directory?

**Answer**: This directory contains data from pytest's cache plugin, which provides the `--lf` (last-failed) and `--ff` (failed-first) options. It is automatically created when running tests and should **NOT** be committed to version control (it's in `.gitignore`).

---

## 🛣️ Roadmap

Future enhancements planned for this project:

- [ ] Add API authentication and rate limiting
- [ ] Integrate MLflow for experiment tracking
- [ ] Implement scheduled retraining pipelines (cron jobs)
- [ ] Build web UI dashboard for predictions
- [ ] Add monitoring and alerting (Prometheus/Grafana)
- [ ] Docker containerization
- [ ] Kubernetes deployment configuration
- [ ] A/B testing framework for model versions

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
- All tests pass (`pytest`)
- New features include appropriate tests
- Update documentation as needed

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

**Project Maintainer**: Sourav Mondal

---

## 🎯 Key Learnings

This project demonstrates:

- ✅ Importance of integrated preprocessing in ML pipelines
- ✅ Preventing feature drift between training and inference
- ✅ Automating ML workflows with CI/CD
- ✅ Proper separation of batch, training, and API layers
- ✅ Building maintainable production ML systems (not just notebooks)
- ✅ Deploying ML models to cloud platforms (Render)
- ✅ Managing model lifecycle from training to deployment

---

## 📊 Prediction Methods Comparison

| Aspect | Batch Predictions | Real-Time API |
|--------|------------------|---------------|
| **Input** | CSV file | JSON request |
| **Output** | CSV file | JSON response |
| **Speed** | Processes many rows | Instant single prediction |
| **Use Case** | Bulk analysis, reports | Live applications, user queries |
| **Trigger** | Manual or CI/CD | HTTP request |
| **Best For** | End-of-day processing | Interactive applications |

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star! ⭐**

**[Try the Live API](https://mlops-cars24-2.onrender.com/docs#/default/home__get)**

</div>