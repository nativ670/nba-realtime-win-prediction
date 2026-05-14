import time
import requests
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playbyplayv3
from src.features.in_game import calculate_in_game_features

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Target API Endpoint
API_URL = "http://127.0.0.1:8000/predict_win_prob"

# Live Game Configuration (Replace with a real active GAME_ID if testing live)
# Example: '0022300001'
GAME_ID = '0022300001' 

# Static Pre-Game Features (Dummy values for testing)
STATIC_FEATURES = {
    "elo_advantage": 45.0,
    "rest_advantage": 1,
    "distance_traveled": 300.0
}

# ==============================================================================
# LIVE POLLER ENGINE
# ==============================================================================

def fetch_latest_pbp(game_id):
    """Fetches the full play-by-play history for the given game ID."""
    try:
        pbp = playbyplayv3.PlayByPlayV3(game_id=game_id)
        df = pbp.get_data_frames()[0]
        return df
    except Exception as e:
        print(f"[!] Error fetching PBP data: {e}")
        return None

def print_scoreboard(period, clock, score_diff, win_prob):
    """Prints a beautiful live scoreboard to the console."""
    # Convert period to Q1, Q2, etc.
    q_str = f"Q{period}" if period <= 4 else f"OT{period-4}"
    
    # Format win prob as percentage
    prob_pct = win_prob * 100
    
    # Score diff sign
    diff_str = f"+{score_diff}" if score_diff > 0 else str(score_diff)
    
    print("-" * 50)
    print(f" LIVE SCOREBOARD | [{q_str} {clock}]")
    print("-" * 50)
    print(f" Score Diff: {diff_str.rjust(3)} | Home Win Prob: {prob_pct:5.1f}%")
    print("-" * 50)

def run_poller(game_id):
    """Main loop for polling live game data and getting predictions."""
    print(f"[*] Starting Live Poller for Game ID: {game_id}")
    print(f"[*] Polling every 15 seconds. Press Ctrl+C to stop.\n")

    while True:
        try:
            # 1. Fetch latest data
            df_pbp = fetch_latest_pbp(game_id)
            
            if df_pbp is not None and not df_pbp.empty:
                # 2. Process features using the engine from src/features/in_game.py
                # This automatically handles score, time, possession, timeouts, and bonus!
                # Since calculate_in_game_features is vectorized for history, it's perfect.
                processed_df = calculate_in_game_features(df_pbp)
                
                # Get the absolute latest state
                latest_state = processed_df.iloc[-1]
                
                # 3. Construct the payload
                payload = {
                    "score_differential": int(latest_state['score_differential']),
                    "seconds_remaining_in_game": float(latest_state['seconds_remaining_in_game']),
                    "possession_team_id": int(latest_state['possession_team_id']) if pd.notna(latest_state['possession_team_id']) else 0,
                    "elo_advantage": float(STATIC_FEATURES['elo_advantage']),
                    "rest_advantage": int(STATIC_FEATURES['rest_advantage']),
                    "distance_traveled": float(STATIC_FEATURES['distance_traveled']),
                    "momentum_differential": float(latest_state['momentum_differential']),
                    "home_timeouts_remaining": int(latest_state['home_timeouts_remaining']),
                    "away_timeouts_remaining": int(latest_state['away_timeouts_remaining']),
                    "home_in_bonus": int(latest_state['home_in_bonus']),
                    "away_in_bonus": int(latest_state['away_in_bonus'])
                }
                
                # 4. Request Win Probability from our FastAPI server
                response = requests.post(API_URL, json=payload)
                
                if response.status_code == 200:
                    win_prob = response.json().get('home_win_probability', 0.5)
                    
                    # 5. Display the scoreboard
                    period = int(latest_state['PERIOD'])
                    clock = latest_state['PCTIMESTRING']
                    score_diff = int(latest_state['score_differential'])
                    
                    print_scoreboard(period, clock, score_diff, win_prob)
                else:
                    print(f"[!] API Error ({response.status_code}): {response.text}")

            else:
                print(f"[?] No data found for Game ID {game_id} yet. Checking again...")

        except requests.exceptions.RequestException as e:
            print(f"[!] Network error: {e}")
        except Exception as e:
            print(f"[!] An unexpected error occurred: {e}")
        
        # 6. Wait before next poll to avoid rate limits
        time.sleep(15)

def run_mock_test():
    """Lightweight test to verify feature calculation and payload formatting."""
    print("[*] Running Mandatory Shootaround (Mock Test)...")
    
    # Create a tiny mock PBP dataframe (3 plays)
    mock_pbp = pd.DataFrame({
        'PERIOD': [1, 1, 1],
        'PCTIMESTRING': ['12:00', '11:30', '11:00'],
        'EVENTMSGTYPE': [10, 1, 1], # Start, Made Shot, Made Shot
        'SCORE': ['0 - 0', '0 - 2', '3 - 2'],
        'PLAYER1_TEAM_ID': [0, 1610612737, 1610612738],
        'EVENTNUM': [1, 2, 3]
    })
    
    try:
        processed_df = calculate_in_game_features(mock_pbp)
        latest_state = processed_df.iloc[-1]
        
        # Verify basic calculations
        assert latest_state['score_differential'] == 1 # 3 - 2
        assert latest_state['seconds_remaining_in_game'] == 2880 - 60 # 12:00 - 1:00
        
        print("[+] Mock Test Passed: Feature engineering logic is sound.")
        return True
    except Exception as e:
        print(f"[!] Mock Test Failed: {e}")
        return False

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # 1. Run mandatory lightweight test
    test_passed = run_mock_test()
    
    if test_passed:
        # 2. Run the real poller (Only if test passes)
        # Note: This will loop forever until interrupted
        try:
            run_poller(GAME_ID)
        except KeyboardInterrupt:
            print("\n[*] Poller stopped by user. Exiting.")
    else:
        print("[!] Poller not started due to test failure.")
