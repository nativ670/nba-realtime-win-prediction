# 🏀 NBA Real-Time Win Probability (WP) Predictor

An end-to-end Machine Learning pipeline that predicts the real-time winning percentage of NBA teams as a game unfolds. Built natively in Python, this project ingests play-by-play data, calculates dynamic in-game features (e.g., score differential, possession, time remaining), and uses an ensemble model (XGBoost + LSTM) to generate predictions.

---

## 🏗️ Architecture & Tech Stack

This project is structured around a fully automated MLOps pipeline:

1. **Data Ingestion (`src/data_ingestion/`)**: 
   - Uses the `nba_api` package to fetch 10+ years of historical Play-By-Play (PBP) and box score data.
   - Merges external historical Elo ratings from Neil Paine's dataset.
   - Automatically computes geographic travel distance using Haversine formulas and days of rest for scheduling advantages.
2. **Feature Engineering (`src/features/`)**: 
   - **Pre-game**: Elo ratings, rest advantage, back-to-back indicators, distance traveled.
   - **In-game**: Score differential, time remaining, possession logic, timeout counts, foul bonus state.
3. **Model Training & Tracking (`src/models/`)**: 
   - **XGBoost Classifier**: Learns tabular state-based win probabilities from game snapshots.
   - **LSTM Neural Network**: Learns sequential game flow and momentum over the trailing 15 actions.
   - **MLflow**: Tracks experiment metrics, hyperparameters, and Expected Calibration Error (ECE).
4. **Live Inference (`src/models/api.py`)**: 
   - **FastAPI**: Serves a highly optimized REST API holding the preloaded models in memory.

**Core Tech Stack**: Python, Pandas/NumPy, Scikit-Learn, XGBoost, TensorFlow/Keras, FastAPI, MLflow.

---

## 🚀 Installation & Setup

Ensure you have Python 3.12+ installed.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/nba-realtime-wp.git
   cd nba-realtime-wp
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🛠️ Usage Guide

### 1. Daily Data Fetching
To pull the latest Elo, RAPTOR, and Play-by-Play data, run the daily update script:
```bash
python src/data_ingestion/daily_update.py
```
*(Note: A GitHub Actions CI/CD workflow is already configured to run this automatically every morning).*

### 2. Training the Models
To train the models and track the metrics in MLflow, run the training scripts:
```bash
# Train the Tabular XGBoost model
python src/models/train.py

# Train the Sequential LSTM model
python src/models/train_nn.py
```
You can view the training metrics by running `mlflow ui` and navigating to `http://localhost:5000`.

### 3. Serving the API
Start the FastAPI inference server:
```bash
uvicorn src.models.api:app --reload
```
Test the ensemble API using the built-in pytest suite:
```bash
pytest tests/
```

---

## 📋 Legal Disclaimer & Data Usage Policy

### Strictly Non-Profit & Educational Use
This project is an open-source, non-profit portfolio piece created strictly for educational and personal data science research. It is not commercialized, monetized, or used for any gambling-related purposes.  

### Not Affiliated with the NBA
This repository and its creator are in no way affiliated with, authorized, maintained, sponsored, or endorsed by the National Basketball Association (NBA) or any of its affiliates or subsidiaries. All NBA logos, team names, and brands are the property of their respective owners.

### Data Sourcing & Fair Use
The factual game data and statistics utilized in this project are in the public domain. Data ingestion is handled via the open-source `nba_api` Python package. Out of respect for the NBA's infrastructure, all automated data polling within this project is strictly rate-limited and run on delayed schedules (e.g., polling once every 24 hours or using 15-second intervals) to ensure zero impact on the league's servers.  

**If you plan to fork this repository, please ensure you do not use this architecture to overwhelm public APIs or for commercial gain.**
