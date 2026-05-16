import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, log_loss, brier_score_loss
import matplotlib.pyplot as plt
import os
import json

# Define features and target constants
BASE_FEATURES = [
    'score_differential', 
    'seconds_remaining_in_game', 
    'possession_team_id', 
    'elo_advantage', 
    'rest_advantage', 
    'distance_traveled'
]

ADVANCED_FEATURES = [
    'momentum_differential', 
    'home_timeouts_remaining', 
    'away_timeouts_remaining', 
    'home_in_bonus', 
    'away_in_bonus',
    'live_raptor_advantage'
]

FEATURES = BASE_FEATURES + ADVANCED_FEATURES
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

    # Use GroupShuffleSplit to prevent game-level leakage
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, test_idx = next(gss.split(df[FEATURES], df[TARGET], groups=df[GROUP_COL]))

    X_train, y_train = df.iloc[train_idx][FEATURES], df.iloc[train_idx][TARGET]
    X_test, y_test = df.iloc[test_idx][FEATURES], df.iloc[test_idx][TARGET]

    print(f"Training set size: {len(X_train)} rows ({df.iloc[train_idx][GROUP_COL].nunique()} games)")
    print(f"Test set size: {len(X_test)} rows ({df.iloc[test_idx][GROUP_COL].nunique()} games)")

    # Initialize and train XGBClassifier
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5
    )

    print("Training model...")
    model.fit(X_train, y_train)
    
    return model, X_test, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluates the model and prints performance metrics."""
    print("Evaluating model...")
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    ll = log_loss(y_test, y_prob)
    bs = brier_score_loss(y_test, y_prob)

    print(f"--- Model Performance ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Log Loss:  {ll:.4f}")
    print(f"Brier Score: {bs:.4f}")
    
    return acc, ll, bs

def plot_and_save_importance(model, output_path='nba_feature_importance.png'):
    """Plots feature importance and saves to file."""
    print(f"Saving feature importance plot to {output_path}...")
    
    plt.figure(figsize=(10, 8))
    xgb.plot_importance(model, importance_type='gain', show_values=False)
    plt.title('Feature Importance (Gain)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_model(model, output_path='src/models/xgb_v3_10man.json'):
    """Saves the trained model to a JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    print(f"Saving model to {output_path}...")
    model.save_model(output_path)

if __name__ == '__main__':
    # Mandatory Shootaround: Lightweight test with mock data
    DATA_PATH = 'data/processed/seasons'
    
    try:
        if os.path.exists(DATA_PATH):
            df = load_data(DATA_PATH)
        else:
            print("Training data not found. Generating mock data for testing...")
            # Create a small mock dataset for testing the script logic
            np.random.seed(42)
            n_games = 5
            rows_per_game = 100
            total_rows = n_games * rows_per_game
            
            mock_data = []
            for game_id in range(n_games):
                for row in range(rows_per_game):
                    mock_data.append({
                        'GAME_ID': f'2023000{game_id}',
                        'score_differential': np.random.randint(-20, 20),
                        'seconds_remaining_in_game': 2880 - (row * 28.8),
                        'possession_team_id': np.random.randint(1610612737, 1610612766),
                        'elo_advantage': np.random.uniform(-100, 100),
                        'rest_advantage': np.random.randint(-2, 3),
                        'distance_traveled': np.random.uniform(0, 2000),
                        'momentum_differential': np.random.uniform(-5, 5),
                        'home_timeouts_remaining': np.random.randint(0, 7),
                        'away_timeouts_remaining': np.random.randint(0, 7),
                        'home_in_bonus': np.random.randint(0, 2),
                        'away_in_bonus': np.random.randint(0, 2),
                        'home_win': 1 if game_id % 2 == 0 else 0 # Simple pattern
                    })
            df = pd.DataFrame(mock_data)

        # Run pipeline
        model, X_test, y_test = train_model(df)
        evaluate_model(model, X_test, y_test)
        plot_and_save_importance(model)
        save_model(model)
        
        print("\nTraining script executed successfully!")

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
