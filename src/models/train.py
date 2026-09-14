import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
import matplotlib.pyplot as plt
import os
import json

from src.config import MODEL_FEATURES as FEATURES
TARGET = 'home_win'
GROUP_COL = 'GAME_ID'

import glob

def load_data(file_path):
    """Loads the processed training data from parquet file(s)."""
    if os.path.isdir(file_path):
        print(f"Loading all partitioned data from {file_path}...")
        files = glob.glob(os.path.join(file_path, "*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in {file_path}")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        return df
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training data not found at {file_path}")
    
    print(f"Loading data from {file_path}...")
    df = pd.read_parquet(file_path)
    return df

def train_model(df):
    """
    Splits data by GAME_ID to prevent leakage and trains an XGBoost classifier.
    """
    print("Preparing data for training...")
    
    # Ensure all features exist in the dataframe
    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataset: {missing_features}")

    # Chronological Split (prevents temporal leakage)
    df = df.sort_values(by=GROUP_COL).reset_index(drop=True)
    unique_games = df[GROUP_COL].unique()
    
    split_idx = int(len(unique_games) * 0.8)
    train_games = set(unique_games[:split_idx])
    
    train_mask = df[GROUP_COL].isin(train_games)
    
    X_train, y_train = df[train_mask][FEATURES], df[train_mask][TARGET]
    X_test, y_test = df[~train_mask][FEATURES], df[~train_mask][TARGET]

    print(f"Training set size: {len(X_train)} rows ({len(train_games)} games)")
    print(f"Test set size: {len(X_test)} rows ({len(unique_games) - len(train_games)} games)")

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'random_state': 42,
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 5
    }

    # Initialize and train XGBClassifier
    model = xgb.XGBClassifier(**params)

    mlflow.log_params(params)

    print("Training model...")
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and prints performance metrics."""
    print("Evaluating model...")
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_prob, labels=[0, 1])
    bs = brier_score_loss(y_test, y_prob)

    print(f"--- Model Performance ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Log Loss:  {ll:.4f}")
    print(f"Brier Score: {bs:.4f}")
    
    # Calculate Expected Calibration Error (ECE) via calibration_curve
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
    ece = np.mean(np.abs(prob_true - prob_pred))
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("log_loss", ll)
    mlflow.log_metric("brier_score", bs)
    mlflow.log_metric("ece", ece)
    
    return acc, ll, bs, ece

def plot_and_save_importance(model, output_path='nba_feature_importance.png'):
    """Plots feature importance and saves to file."""
    print(f"Saving feature importance plot to {output_path}...")
    
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model, importance_type='gain', show_values=False)
    plt.title('Feature Importance (Gain)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

from src.config import XGB_MODEL_PATH
def save_model(model, output_path=str(XGB_MODEL_PATH)):
    """Saves the trained model to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving model to {output_path}...")
    model.save_model(output_path)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-only", action="store_true", help="Run a fast mock shootaround test")
    args = parser.parse_args()

    # Setup MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("NBA_RealTime_WP")

    from src.config import SEASONS_DIR
    DATA_PATH = str(SEASONS_DIR)
    
    try:
        if args.test_only or not os.path.exists(DATA_PATH):
            print("Running in test mode. Generating mock data for testing...")
            np.random.seed(42)
            n_games = 5
            rows_per_game = 100
            
            mock_data = []
            for game_id in range(n_games):
                for row in range(rows_per_game):
                    mock_data.append({
                        'GAME_ID': f'2023000{game_id}',
                        'season': 2023,
                        'score_differential': np.random.randint(-20, 20),
                        'seconds_remaining_in_game': 2880 - (row * 28.8),
                        'is_home_possession': float(np.random.choice([0.0, 1.0])),
                        'elo_advantage': np.random.uniform(-100, 100),
                        'rest_advantage': np.random.randint(-2, 3),
                        'distance_traveled': np.random.uniform(0, 2000),
                        'momentum_differential': np.random.uniform(-5, 5),
                        'home_timeouts_remaining': np.random.randint(0, 7),
                        'away_timeouts_remaining': np.random.randint(0, 7),
                        'home_in_bonus': np.random.randint(0, 2),
                        'away_in_bonus': np.random.randint(0, 2),
                        'live_raptor_advantage': np.random.uniform(-5, 5),
                        'home_win': 1 if game_id % 2 == 0 else 0
                    })
            df = pd.DataFrame(mock_data)
        else:
            print(f"Loading real data from {DATA_PATH}...")
            df = load_data(DATA_PATH)

        # Run pipeline
        # Avoid creating random MLflow runs during test-only mock runs
        if args.test_only:
            print("Mock mode: skipping MLflow tracking.")
            model, X_test, y_test = train_model(df)
            evaluate_model(model, X_test, y_test)
            print("Mock test completed successfully!")
            import sys
            sys.exit(0)
            
        with mlflow.start_run():
            model, X_test, y_test = train_model(df)
            evaluate_model(model, X_test, y_test)
            plot_and_save_importance(model)
            mlflow.log_artifact('nba_feature_importance.png')
            save_model(model)
            mlflow.xgboost.log_model(model, "xgboost-model")
        
        print("\nTraining script executed successfully!")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
