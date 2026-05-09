# Superconductivity Critical Temperature Predictor

An end-to-end machine learning system designed to predict the critical temperature ($T_c$) of superconducting materials based on 81 chemical and physical features. This project leverages **SVR (Support Vector Regression)** optimized via **Optuna** and serves predictions through a containerized **FastAPI** and **Streamlit** ecosystem.

---

## 🛠 Tech Stack

*   **Language:** Python 3.11
*   **Machine Learning:** Scikit-learn (SVR), PCA, Optuna (Hyperparameter Optimization)
*   **Experiment Tracking:** MLflow (Self-hosted with SQLite backend)
*   **API Framework:** FastAPI (Uvicorn)
*   **Frontend:** Streamlit
*   **Infrastructure:** Docker & Docker Compose, WSL2 (Ubuntu)
*   **Data Source:** UCI Superconductivity Dataset

---

## 🏗 System Components

### 1. Training Pipeline (`run_pipy`)
*   Implements a `StandardScaler` → `PCA` → `SVR` pipeline.
*   Uses **Optuna** to minimize RMSE by tuning $C$, $\epsilon$, and PCA components.
*   Logs all parameters, metrics (RMSE, MAE, $R^2$), and the final model artifact to a local MLflow server.

### 2. MLflow Server
*   Acts as the central model registry.
*   Configured with a persistent SQLite database and local artifact storage to survive container restarts.

### 3. Prediction API (`main.py`)
*   A FastAPI wrapper that pulls the `latest` model version from MLflow on startup.
*   Validates input data using Pydantic.
*   **Feature Alignment:** Automatically re-attaches feature names to incoming vectors to ensure consistency with the training scaler.

### 4. User Interface (`ui.py`)
*   **Manual Entry:** Predict $T_c$ for a single material.
*   **Batch Upload:** Supports CSV uploads for bulk analysis.
*   **Model Diagnostics:** Includes a "Health Check" to verify connection to the MLflow backend.

---

## 🚀 How to Run

### Prerequisites
*   Docker and Docker Compose installed.
*   The dataset CSV located in the project root.

### 1. Start the Infrastructure
Launch the MLflow tracking server and the storage volumes:
```bash
docker compose up -d

### 2. access port:5000 for the ui, 5001 for the api, and 5002 for ML-Flow