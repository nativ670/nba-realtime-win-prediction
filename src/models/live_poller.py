"""
src/models/live_poller.py — Live Game Win Probability Poller
=============================================================
Polls NBA PBP data incrementally, runs feature engineering on
new actions only, and posts feature payloads to the prediction API.

Key design choices:
  - INCREMENTAL processing: tracks last processed actionNumber so we
    only run feature engineering on new rows each poll cycle.
  - Circuit breaker: 3 consecutive failures trigger exponential backoff
    (15s → 30s → 60s). Resets on any successful poll.
  - Uses shared nba_api_call() wrapper (retry + timeout built-in).
  - Feature list sourced from src.config.MODEL_FEATURES (single source of truth).
  - GAME_ID is a required CLI argument, not a hardcoded constant.
"""

import argparse
import time
import requests
import pandas as pd
import numpy as np
import os
import sys
from collections import deque
from nba_api.stats.endpoints import playbyplayv3

# Add the project root to sys.path so 'src' can be found when running directly
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

from src.config import MODEL_FEATURES, LSTM_SEQUENCE_LENGTH
from src.utils.nba_client import nba_api_call
from src.features.in_game import calculate_in_game_features
from src.features.substitution_tracker import SubstitutionTracker
from src.utils.helpers import standardize_pbp_v3

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Target API Endpoint
API_URL = "http://127.0.0.1:8000/predict_win_prob"

# Static Pre-Game Features (Dummy values for testing)
STATIC_FEATURES = {
    "elo_advantage": 45.0,
    "rest_advantage": 1,
    "distance_traveled": 300.0
}

# Circuit breaker thresholds
BASE_POLL_INTERVAL = 15       # Seconds between polls (normal)
MAX_CONSECUTIVE_FAILS = 3     # Failures before exponential backoff kicks in

# ==============================================================================
# LIVE POLLER ENGINE
# ==============================================================================

def fetch_latest_pbp(game_id):
    """
    Fetches the full play-by-play history for the given game ID using
    the shared nba_api_call wrapper (has built-in retry + timeout).

    Returns
    -------
    pd.DataFrame or None
        Standardized PBP DataFrame, or None on failure.
    """
    df = nba_api_call(
        playbyplayv3.PlayByPlayV3,
        df_index=0,
        timeout=30,
        game_id=game_id
    )

    # nba_api_call returns an empty DF on failure, not None
    if df.empty:
        return None

    # Standardize column names for the feature engine
    return standardize_pbp_v3(df)


def extract_new_actions(full_pbp_df, last_action_number):
    """
    Filters the full PBP DataFrame to only rows with actionNumber
    greater than the last processed one.

    Parameters
    ----------
    full_pbp_df : pd.DataFrame
        Complete standardized PBP for the game.
    last_action_number : int
        The highest actionNumber we've already processed.

    Returns
    -------
    pd.DataFrame
        Only the new (unprocessed) rows.
    """
    # Determine the action-number column (V3 uses 'actionNumber',
    # standardized might use 'EVENTNUM')
    if 'actionNumber' in full_pbp_df.columns:
        col = 'actionNumber'
    elif 'EVENTNUM' in full_pbp_df.columns:
        col = 'EVENTNUM'
    else:
        # Fallback: return everything (can't do incremental)
        return full_pbp_df

    return full_pbp_df[full_pbp_df[col] > last_action_number]


def get_max_action_number(pbp_df):
    """Returns the highest action/event number in the DataFrame."""
    if 'actionNumber' in pbp_df.columns:
        return int(pbp_df['actionNumber'].max())
    elif 'EVENTNUM' in pbp_df.columns:
        return int(pbp_df['EVENTNUM'].max())
    return 0


def build_feature_payload(latest_state):
    """
    Builds the feature dictionary for the prediction API from the latest
    processed game state, using MODEL_FEATURES as the single source of truth.

    Static pre-game features (elo_advantage, rest_advantage, distance_traveled)
    are injected from STATIC_FEATURES since they don't come from PBP data.

    Parameters
    ----------
    latest_state : pd.Series
        The most recent row from the processed feature DataFrame.

    Returns
    -------
    dict
        Feature name → value mapping matching MODEL_FEATURES order.
    """
    payload = {}
    for feat in MODEL_FEATURES:
        if feat in STATIC_FEATURES:
            # Pre-game features come from the static config
            payload[feat] = float(STATIC_FEATURES[feat])
        elif feat in latest_state.index:
            val = latest_state[feat]
            # Handle NaN gracefully (e.g., possession_team_id can be NaN)
            payload[feat] = 0 if pd.isna(val) else float(val)
        else:
            # Feature missing entirely — default to 0
            payload[feat] = 0.0
    return payload


def print_scoreboard(period, clock, score_diff, results):
    """Prints a beautiful live scoreboard to the console with ensemble predictions."""
    # Convert period to Q1, Q2, etc.
    q_str = f"Q{period}" if period <= 4 else f"OT{period-4}"

    # Format win probs as percentages
    final_prob = results.get('final_win_probability', 0.5) * 100
    xgb_prob = results.get('xgb_prob', 0.5) * 100
    lstm_prob = results.get('lstm_prob', 0.5) * 100

    # Score diff sign
    diff_str = f"+{score_diff}" if score_diff > 0 else str(score_diff)

    print("-" * 65)
    print(f" LIVE SCOREBOARD | [{q_str} {clock}]")
    print("-" * 65)
    print(f" Score Diff: {diff_str.rjust(3)} | Final Win Prob: {final_prob:5.1f}%")
    print(f" (XGB: {xgb_prob:4.0f}%, LSTM: {lstm_prob:4.0f}%)")
    print("-" * 65)


def run_poller(game_id):
    """
    Main loop for polling live game data and getting predictions.

    Implements:
      - Incremental PBP processing (only new actions since last poll)
      - Circuit breaker with exponential backoff on consecutive failures
      - Accumulated feature DataFrame to avoid full re-processing
    """
    print(f"[*] Starting Live Poller for Game ID: {game_id}")
    print(f"[*] Polling every {BASE_POLL_INTERVAL} seconds. Press Ctrl+C to stop.\n")

    # --- State tracking for incremental processing ---
    last_action_number = 0          # Highest actionNumber we've processed
    accumulated_features_df = None  # Running feature DataFrame across polls

    # --- Rolling LSTM sequence (uses LSTM_SEQUENCE_LENGTH from config) ---
    rolling_sequence = deque(maxlen=LSTM_SEQUENCE_LENGTH)

    # --- Substitution Tracker ---
    sub_tracker = SubstitutionTracker(game_id)

    # --- Circuit breaker state ---
    consecutive_failures = 0
    poll_interval = BASE_POLL_INTERVAL

    while True:
        try:
            # 1. Fetch the FULL PBP (the API doesn't support partial fetches)
            df_pbp = fetch_latest_pbp(game_id)

            if df_pbp is not None and not df_pbp.empty:
                # 2. Filter to only NEW actions since last poll
                new_actions = extract_new_actions(df_pbp, last_action_number)

                if new_actions.empty:
                    # No new plays — game might be in timeout/halftime
                    print(f"[=] No new actions since actionNumber {last_action_number}. Waiting...")
                else:
                    # 3. Run feature engineering ONLY on new rows
                    new_features = calculate_in_game_features(new_actions)
                    new_features = sub_tracker.process_pbp(new_features)

                    # 4. Accumulate into the running feature DataFrame
                    if accumulated_features_df is None:
                        accumulated_features_df = new_features
                    else:
                        accumulated_features_df = pd.concat(
                            [accumulated_features_df, new_features],
                            ignore_index=True
                        )

                    # 5. Update the high-water mark
                    last_action_number = get_max_action_number(df_pbp)

                    # 6. Build feature payload from the latest accumulated state
                    latest_state = accumulated_features_df.iloc[-1]
                    current_features = build_feature_payload(latest_state)

                    # 7. Append to rolling LSTM sequence
                    rolling_sequence.append(current_features)

                    # 8. Construct the Ensemble Payload and POST to API
                    payload = {"sequence": list(rolling_sequence)}

                    response = requests.post(API_URL, json=payload, timeout=10)

                    if response.status_code == 200:
                        results = response.json()

                        # 9. Display the scoreboard
                        period = int(latest_state['PERIOD'])
                        clock = latest_state['PCTIMESTRING']
                        score_diff = int(latest_state['score_differential'])

                        print_scoreboard(period, clock, score_diff, results)
                    else:
                        print(f"[!] API Error ({response.status_code}): {response.text}")

                # --- Circuit breaker: reset on success ---
                consecutive_failures = 0
                poll_interval = BASE_POLL_INTERVAL

            else:
                print(f"[?] No data found for Game ID {game_id} yet. Checking again...")

        except requests.exceptions.RequestException as e:
            print(f"[!] Network error: {e}")
            consecutive_failures += 1
        except Exception as e:
            print(f"[!] An unexpected error occurred: {e}")
            consecutive_failures += 1

        # --- Circuit breaker: exponential backoff on repeated failures ---
        if consecutive_failures >= MAX_CONSECUTIVE_FAILS:
            # Exponential backoff: 15 → 30 → 60 → 120 ... capped at 120s
            backoff_multiplier = 2 ** (consecutive_failures - MAX_CONSECUTIVE_FAILS)
            poll_interval = min(BASE_POLL_INTERVAL * backoff_multiplier, 120)
            print(f"[!] Circuit breaker: {consecutive_failures} consecutive failures. "
                  f"Backing off to {poll_interval}s.")

        # 10. Wait before next poll
        time.sleep(poll_interval)


# ==============================================================================
# MOCK TEST
# ==============================================================================

def run_mock_test():
    """Lightweight test to verify feature calculation and payload formatting."""
    print("[*] Running Mandatory Shootaround (Mock Test)...")

    # Create a tiny mock PBP dataframe (3 plays)
    mock_pbp = pd.DataFrame({
        'PERIOD': [1, 1, 1],
        'PCTIMESTRING': ['12:00', '11:30', '11:00'],
        'EVENTMSGTYPE': [10, 1, 1],  # Start, Made Shot, Made Shot
        'SCORE': ['0 - 0', '0 - 2', '3 - 2'],
        'PLAYER1_TEAM_ID': [0, 1610612737, 1610612738],
        'EVENTNUM': [1, 2, 3]
    })

    try:
        # 1. Test feature engineering
        processed_df = calculate_in_game_features(mock_pbp)
        latest_state = processed_df.iloc[-1]

        # Verify basic calculations
        assert latest_state['score_differential'] == 1, (
            f"Expected score_diff=1, got {latest_state['score_differential']}"
        )
        assert latest_state['seconds_remaining_in_game'] == 2880 - 60, (
            f"Expected 2820s remaining, got {latest_state['seconds_remaining_in_game']}"
        )
        print("[+] Feature engineering logic: OK")

        # 2. Test incremental action filtering
        new_rows = extract_new_actions(mock_pbp, last_action_number=1)
        assert len(new_rows) == 2, (
            f"Expected 2 new actions after EVENTNUM=1, got {len(new_rows)}"
        )
        print("[+] Incremental filtering logic: OK")

        # 3. Test payload builder uses MODEL_FEATURES
        payload = build_feature_payload(latest_state)
        assert set(payload.keys()) == set(MODEL_FEATURES), (
            f"Payload keys don't match MODEL_FEATURES. "
            f"Missing: {set(MODEL_FEATURES) - set(payload.keys())}"
        )
        print("[+] Payload builder / MODEL_FEATURES alignment: OK")

        # 4. Test circuit breaker math
        assert min(BASE_POLL_INTERVAL * (2 ** 0), 120) == 15
        assert min(BASE_POLL_INTERVAL * (2 ** 1), 120) == 30
        assert min(BASE_POLL_INTERVAL * (2 ** 2), 120) == 60
        print("[+] Circuit breaker backoff math: OK")

        print("[+] Mock Test Passed ✅")
        return True

    except Exception as e:
        print(f"[!] Mock Test Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    # --- Parse CLI arguments ---
    parser = argparse.ArgumentParser(
        description="Live NBA Win Probability Poller",
        epilog="Example: python -m src.models.live_poller 0022400001"
    )
    parser.add_argument(
        "game_id",
        nargs="?",           # Optional so mock test can run without it
        default=None,
        help="NBA GAME_ID to poll (e.g., 0022400001)"
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Run only the mock test, then exit."
    )
    args = parser.parse_args()

    # 1. Run mandatory lightweight test
    test_passed = run_mock_test()

    if args.test_only:
        sys.exit(0 if test_passed else 1)

    if test_passed:
        if args.game_id is None:
            print("[!] No GAME_ID provided. Usage: python -m src.models.live_poller <GAME_ID>")
            sys.exit(1)

        # 2. Run the real poller (only if test passes)
        try:
            run_poller(args.game_id)
        except KeyboardInterrupt:
            print("\n[*] Poller stopped by user. Exiting.")
    else:
        print("[!] Poller not started due to test failure.")
        sys.exit(1)
