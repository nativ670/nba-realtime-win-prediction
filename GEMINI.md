# PROJECT PLAYBOOK: NBA Real-Time Win Probability (WP) Predictor

## 🏀 The Mission
We are building a machine learning project to predict the real-time winning percentage of NBA teams as the game unfolds. The model will use a pregame baseline (Elo ratings) and update dynamically based on live in-game features. The model will be trained on the last 10 years of historical NBA play-by-play data.

## 🛠️ The Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Data Sources:** `nba_api` (for box scores/play-by-play), Neil Paine's GitHub (for Elo)
* **Machine Learning:** Scikit-Learn, XGBoost

## 📊 The Feature Engineering Game Plan
This project is a migration from an existing R script into Python. Do not start from scratch on the logic; adapt the known winning formulas!
* **Pre-Game Features:** * Elo Ratings (fetched dynamically from Neil Paine's dataset).
    * Rest Advantage (days of rest for team vs. opponent, capped at 5 days).
    * Back-to-Back indicators (Game 1 vs. Game 2 of a B2B).
    * Travel Distance (calculated via Haversine formula using venue coordinates).
* **In-Game Features (Real-Time Engine):**
    * Score Differential (Home Score - Away Score).
    * Time Remaining in the quarter/game.
    * Possession (Who has the ball).
    * Timeouts Remaining.

## 📁 The Directory Structure
The project is strictly organized as follows. All new code must be placed in the appropriate directory:
* `data/` (raw, processed, external)
* `notebooks/` (Jupyter notebooks for EDA and testing)
* `src/data_ingestion/` (Scripts like fetch_pbp.py, fetch_elo.py)
* `src/features/` (Scripts like pregame.py, in_game.py)
* `src/models/` (Scripts like train.py, predict.py)
* `src/utils/` (Helper functions, coordinate dictionaries)

## 🧠 Coding Guidelines for Claude
* Write clean, modular, and highly commented Python code.
* Assume the user is an expert in data science but is transitioning this specific codebase from R to Python.
* Prioritize Pandas vectorization over loops for all feature engineering.
* Always include robust error handling, especially when hitting web APIs for data.
* Remember to update the requirements.txt file when necessary.
* **Clean Code:** Write clean, modular, and highly commented Python code, prioritizing Pandas vectorization over loops.
* **Storage Efficiency (No Bloat):** Do NOT save massive raw datasets (like 10 years of CSVs) to the local machine. Keep local storage to an absolute minimum. Process data in memory when possible. If data absolutely must be cached locally, use compressed formats like `.parquet` instead of bulky `.csv` files.
* **Optimized Fetching (The Fast Break):** Make data fetching as efficient and lightweight as possible. Utilize intelligent batching, caching, or asynchronous requests to avoid redundant API calls while strictly respecting the `nba_api` rate limits.
* **Built-in Shootarounds (Mandatory Testing):** EVERY single Python script must include an `if __name__ == "__main__":` block at the bottom. This block must execute a lightweight test of the script's functions using a tiny, mocked, or severely limited subset of data (e.g., just 1 game or 1 season). We must be able to run and verify the code on a small scale before applying it to the massive 10-year dataset.