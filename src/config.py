"""
src/config.py — Centralized Project Configuration
===================================================
Single source of truth for all paths, constants, API settings,
and model feature definitions. Import from here instead of
hardcoding values in individual scripts.
"""

import os
from pathlib import Path
from datetime import datetime

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================

# Project root is two levels up from this file (src/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EXTERNAL_DIR = DATA_DIR / "external"
SEASONS_DIR = PROCESSED_DIR / "seasons"

# Key data file paths
LIVE_ELO_PATH = PROCESSED_DIR / "live_elo.parquet"
RAPTOR_PATH = PROCESSED_DIR / "current_raptor.csv"
RAW_PBP_PATH = RAW_DIR / "nba_pbp_10y.parquet"
RAW_BOX_PATH = RAW_DIR / "nba_box_scores_10y.parquet"

# Model file paths
MODELS_DIR = PROJECT_ROOT / "src" / "models"
XGB_MODEL_PATH = MODELS_DIR / "xgb_v3_10man.json"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_v3_10man.keras"

# ==============================================================================
# SEASON / YEAR HELPERS
# ==============================================================================

def get_current_season_year():
    """
    Returns the NBA season year (e.g., 2025 for the 2024-25 season).
    NBA seasons span Oct-Jun, so Oct-Dec belong to the NEXT year's season.
    """
    now = datetime.now()
    return now.year + 1 if now.month >= 10 else now.year


def get_current_season_string():
    """
    Returns the NBA season string (e.g., '2024-25' for the 2024-25 season).
    """
    year = get_current_season_year()
    return f"{year - 1}-{str(year)[2:]}"


def get_season_year_from_game_id(game_id: str) -> int:
    """
    Extracts the season year from an NBA GAME_ID.
    Game IDs encode the season in characters 3-4 (e.g., '0022400001' -> season 2024-25 -> 2025).
    """
    season_code = int(game_id[3:5])
    return 2000 + season_code if season_code < 50 else 1900 + season_code


# ==============================================================================
# ELO CONFIGURATION
# ==============================================================================

K_FACTOR = 20
HOME_ADVANTAGE = 100
OFFSEASON_REGRESSION_FACTOR = 0.75
OFFSEASON_MEAN = 1505  # Slightly above 1500 to account for expansion dilution

# External Elo data source (Neil Paine's dataset)
ELO_SOURCE_URL = "https://raw.githubusercontent.com/Neil-Paine-1/NBA-elo/main/nba_elo.csv"

# ==============================================================================
# NBA API SETTINGS
# ==============================================================================

# Rate limiting — nba_api throttles aggressively on shared IPs (e.g., GitHub Actions)
NBA_API_SLEEP = 0.6           # Seconds between routine calls
NBA_API_RETRY_SLEEP = 5       # Initial backoff on retry (scales: 5s, 10s, 15s)
NBA_API_MAX_RETRIES = 3       # Max retry attempts
NBA_API_TIMEOUT = 60          # Seconds before timeout (increased for slow endpoints)

# Daily update — how many days back to check for new games
DAILY_UPDATE_LOOKBACK_DAYS = 3

# ==============================================================================
# RAPTOR WEIGHTS (Box-Score Prior Regression Coefficients)
# ==============================================================================
# Estimated from Neil Paine's RAPTOR methodology.
# Used by both raptor_engine.py and generate_missing_raptor.py.

RAPTOR_OFF_WEIGHTS = {
    'intercept': -3.88704,
    'MPG': 0.026112,
    'PTS': 0.662784,
    'TSA': -0.51622,
    'AST': 0.430454,
    'TOV': -0.893465,
    'ORB': 0.303023,
    'DRB': -0.085637,
    'STL': 0.418092,
    'BLK': -0.230734,
    'PF': -0.108369
}

RAPTOR_DEF_WEIGHTS = {
    'intercept': -3.079144,
    'MPG': 0.033637,
    'PTS': -0.081412,
    'TSA': 0.025422,
    'AST': -0.025109,
    'TOV': -0.055809,
    'ORB': -0.099034,
    'DRB': 0.191569,
    'STL': 1.150891,
    'BLK': 0.611107,
    'PF': 0.010649
}

# ==============================================================================
# MODEL FEATURES (Single Source of Truth)
# ==============================================================================
# The exact feature list in the exact order used for model training/prediction.
# Any change here must be reflected in model retraining.

MODEL_FEATURES = [
    'score_differential',
    'seconds_remaining_in_game',
    'is_home_possession',
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

# Ensemble blend weights
XGB_WEIGHT = 0.70
LSTM_WEIGHT = 0.30
LSTM_SEQUENCE_LENGTH = 15

# ==============================================================================
# MAIN (Self-Test)
# ==============================================================================

if __name__ == "__main__":
    print("=== NBA Project Config Self-Test ===")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"LIVE_ELO_PATH: {LIVE_ELO_PATH}")
    print(f"Current Season: {get_current_season_string()} (year={get_current_season_year()})")
    print(f"Season from GAME_ID '0022400123': {get_season_year_from_game_id('0022400123')}")
    print(f"Model Features ({len(MODEL_FEATURES)}): {MODEL_FEATURES}")
    print(f"Paths exist:")
    print(f"  DATA_DIR: {DATA_DIR.exists()}")
    print(f"  SEASONS_DIR: {SEASONS_DIR.exists()}")
    print(f"  LIVE_ELO: {LIVE_ELO_PATH.exists()}")
    print("Config OK ✅")
