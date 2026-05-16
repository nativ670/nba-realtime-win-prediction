import pandas as pd
import numpy as np
import xgboost as xgb
import os
import time

# Feature list must EXACTLY match the order used in train.py
FEATURES = [
    'score_differential', 
    'seconds_remaining_in_game', 
    'possession_team_id', 
    'elo_advantage', 
    'rest_advantage', 
    'distance_traveled',
    'momentum_differential', 
    'home_timeouts_remaining', 
    'away_timeouts_remaining', 
    'home_in_bonus', 
    'away_in_bonus',
    'live_raptor_advantage'
]

MODEL_PATH = 'src/models/xgb_v3_10man.json'
SEASONS_DIR = 'data/processed/seasons'

def load_inference_model(model_path):
    """Loads the trained XGBoost model for prediction."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run train.py first.")
    
    print(f"Loading model from {model_path}...")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    return model

def run_live_simulator(game_id, delay=0.1):
    """
    Simulates a live game feed using historical data and provides real-time win probability.
    """
    # 1. Load Model
    model = load_inference_model(MODEL_PATH)
    
    # 2. Determine season and load data
    season_suffix = int(game_id[3:5])
    season_year = 2000 + season_suffix if season_suffix < 50 else 1900 + season_suffix
    data_path = os.path.join(SEASONS_DIR, f"pbp_{season_year}.parquet")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data for season {season_year} not found at {data_path}")

    print(f"Loading data from {data_path} and filtering for Game ID: {game_id}...")
    df = pd.read_parquet(data_path)
    game_data = df[df['GAME_ID'] == game_id].copy()
    
    if game_data.empty:
        raise ValueError(f"No data found for Game ID: {game_id}")
    
    # 3. Sort chronologically (Descending seconds_remaining)
    game_data = game_data.sort_values('seconds_remaining_in_game', ascending=False)
    
    print(f"\n--- STARTING LIVE SIMULATION: GAME {game_id} ---")
    print("-" * 70)
    
    # 4. Iterate and Predict
    for _, row in game_data.iterrows():
        # Extract features in the correct order
        # Convert to numpy and reshape for a single sample prediction
        feature_vector = row[FEATURES].values.reshape(1, -1)
        
        # Get Home Win Probability
        # predict_proba returns [prob_class_0, prob_class_1]
        probs = model.predict_proba(feature_vector)
        home_win_prob = probs[0][1]
        
        # Format display variables
        time_left = int(row['seconds_remaining_in_game'])
        score_diff = int(row['score_differential'])
        possession = "Home" if row['possession_team_id'] == row['possession_team_id'] else "Away" # Mock logic for display if needed
        # In our data, possession_team_id is the actual team ID. 
        # For simplicity in output, we'll just show the score diff and prob.
        
        # Determine score diff sign for formatting
        diff_str = f"+{score_diff}" if score_diff > 0 else str(score_diff)
        
        # Print update
        print(f"[Time Left: {time_left:4d}s] Score Diff: {diff_str:4s} | Home Win Prob: {home_win_prob:.1%}")
        
        # Small delay to simulate "live" feel
        time.sleep(delay)

    print("-" * 70)
    actual_winner = "Home" if game_data.iloc[-1]['home_win'] == 1 else "Away"
    print(f"SIMULATION COMPLETE. Actual Winner: {actual_winner}")

if __name__ == "__main__":
    # Selected a close game identified in research
    TEST_GAME_ID = '0021500391' 
    
    try:
        # Run with a small delay for readability
        run_live_simulator(TEST_GAME_ID, delay=0.05)
    except Exception as e:
        print(f"Error during simulation: {e}")
