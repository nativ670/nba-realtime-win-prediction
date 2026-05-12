import pandas as pd
import numpy as np
import re

# ==============================================================================
# FEATURE ENGINEERING ENGINE
# ==============================================================================

def calculate_in_game_features(df_pbp):
    """
    Transforms raw NBA Play-by-Play data into real-time features.
    Optimized with Pandas vectorization for processing millions of rows.
    
    New Features: 
    - home_points_last_3_mins
    - away_points_last_3_mins
    - momentum_differential
    """
    df = df_pbp.copy()

    # 1. TIME CALCULATIONS
    # Convert 'PERIOD' and 'PCTIMESTRING' to seconds remaining and elapsed time
    
    clock_parts = df['PCTIMESTRING'].str.split(':', expand=True).astype(float)
    df['seconds_remaining_in_period'] = (clock_parts[0] * 60) + clock_parts[1]

    # Seconds remaining in game (countdown)
    df['seconds_remaining_in_game'] = np.where(
        df['PERIOD'] <= 4,
        (4 - df['PERIOD']) * 720 + df['seconds_remaining_in_period'],
        df['seconds_remaining_in_period'] 
    )

    # Elapsed time from start (increasing) - Essential for rolling windows
    # Regulation: 4 periods of 12 mins (720s). OT: 5 mins (300s).
    df['elapsed_time'] = np.where(
        df['PERIOD'] <= 4,
        (df['PERIOD'] - 1) * 720 + (720 - df['seconds_remaining_in_period']),
        2880 + (df['PERIOD'] - 5) * 300 + (300 - df['seconds_remaining_in_period'])
    )

    # 2. SCORE DIFFERENTIAL
    # Extract scores, handling rows with no score update (NaN)
    # Raw PBP format for 'SCORE' is typically "AWAY - HOME" (e.g., "102 - 105")
    scores = df['SCORE'].str.split(' - ', expand=True)
    df['home_score'] = pd.to_numeric(scores[1], errors='coerce')
    df['away_score'] = pd.to_numeric(scores[0], errors='coerce')

    # Forward fill to ensure every row has the current score
    df[['home_score', 'away_score']] = df[['home_score', 'away_score']].ffill().fillna(0)
    df['score_differential'] = df['home_score'] - df['away_score']

    # 3. POSSESSION TRACKING
    team_ids = df['PLAYER1_TEAM_ID'].unique()
    team_ids = [tid for tid in team_ids if pd.notna(tid) and tid != 0]
    
    df['possession_team_id'] = np.nan

    if len(team_ids) >= 2:
        t1, t2 = team_ids[0], team_ids[1]
        other_team_map = {t1: t2, t2: t1}
        
        # Rule 1: Rebound (Event 4) -> Possession goes to the rebounder
        mask_reb = df['EVENTMSGTYPE'] == 4
        df.loc[mask_reb, 'possession_team_id'] = df.loc[mask_reb, 'PLAYER1_TEAM_ID']
        
        # Rule 2: Make (Event 1) or Turnover (Event 5) -> Possession flips to the other team
        mask_flip = df['EVENTMSGTYPE'].isin([1, 5])
        df.loc[mask_flip, 'possession_team_id'] = df.loc[mask_flip, 'PLAYER1_TEAM_ID'].map(other_team_map)
    
    df['possession_team_id'] = df['possession_team_id'].ffill()

    # 4. MOMENTUM FEATURES (Lookback 180s)
    # We use pd.merge_asof to find the score 3 minutes ago
    # merge_asof requires sorting by the key (elapsed_time)
    df = df.sort_values('elapsed_time')
    df['time_lookback'] = df['elapsed_time'] - 180
    df['temp_idx'] = range(len(df)) # Preserve original sequence for stability
    
    # Check for GAME_ID to allow grouped merge_asof (if processing multiple games at once)
    group_col = 'GAME_ID' if 'GAME_ID' in df.columns else None
    
    df_momentum = pd.merge_asof(
        df,
        df[['elapsed_time', 'home_score', 'away_score'] + ([group_col] if group_col else [])],
        left_on='time_lookback',
        right_on='elapsed_time',
        by=group_col,
        direction='backward', # State at or before lookback time
        suffixes=('', '_hist')
    )
    
    # Fill historical scores for the beginning of the game (lookback < 0)
    df_momentum['home_score_hist'] = df_momentum['home_score_hist'].fillna(0)
    df_momentum['away_score_hist'] = df_momentum['away_score_hist'].fillna(0)
    
    # Calculate Momentum columns
    df_momentum['home_points_last_3_mins'] = df_momentum['home_score'] - df_momentum['home_score_hist']
    df_momentum['away_points_last_3_mins'] = df_momentum['away_score'] - df_momentum['away_score_hist']
    df_momentum['momentum_differential'] = df_momentum['home_points_last_3_mins'] - df_momentum['away_points_last_3_mins']
    
    # Restore order and drop temp columns
    cols_to_drop = ['elapsed_time', 'time_lookback', 'temp_idx', 'elapsed_time_hist', 'home_score_hist', 'away_score_hist']
    df = df_momentum.sort_values('temp_idx').drop(columns=cols_to_drop)

    return df

# ==============================================================================
# MAIN EXECUTION (LIGHTWEIGHT TEST)
# ==============================================================================

if __name__ == "__main__":
    print("RUNNING LIGHTWEIGHT IN-GAME FEATURE TEST...")
    
    # Mock Play-by-Play Data (Home: 1610612744, Away: 1610612738)
    # Score format in raw PBP is typically "AWAY - HOME"
    mock_pbp = pd.DataFrame({
        'EVENTNUM': [1, 2, 3, 4, 5, 6],
        'PERIOD': [1, 1, 1, 1, 1, 1],
        'PCTIMESTRING': ['12:00', '11:00', '9:00', '8:50', '8:00', '7:00'],
        'EVENTMSGTYPE': [10, 1, 1, 1, 1, 1],
        'SCORE': [None, '0 - 2', '0 - 5', '2 - 5', '4 - 5', '4 - 10'],
        'PLAYER1_TEAM_ID': [None, 1610612744, 1610612744, 1610612738, 1610612738, 1610612744],
    })

    processed_df = calculate_in_game_features(mock_pbp)
    
    print("\nProcessed Features Sample:")
    cols_to_show = ['PCTIMESTRING', 'home_score', 'away_score', 'home_points_last_3_mins', 'away_points_last_3_mins', 'momentum_differential']
    print(processed_df[cols_to_show])
    
    # Momentum Validations
    
    # 1. T=8:00 (Elapsed 240s). 3 mins ago was T=11:00 (Elapsed 60s).
    # Current Score: Home 5, Away 4.
    # Score at T=11:00 was: Home 2, Away 0.
    # Expected Home Pts Last 3m: 5 - 2 = 3.
    # Expected Away Pts Last 3m: 4 - 0 = 4.
    # Expected Momentum Diff: 3 - 4 = -1.
    row_800 = processed_df[processed_df['PCTIMESTRING'] == '8:00'].iloc[0]
    assert row_800['home_points_last_3_mins'] == 3, f"Expected 3, got {row_800['home_points_last_3_mins']}"
    assert row_800['away_points_last_3_mins'] == 4, f"Expected 4, got {row_800['away_points_last_3_mins']}"
    assert row_800['momentum_differential'] == -1, f"Expected -1, got {row_800['momentum_differential']}"
    
    # 2. T=7:00 (Elapsed 300s). 3 mins ago was T=10:00 (Elapsed 120s).
    # Last event before T=10:00 was T=11:00 (Home 2, Away 0).
    # Current Score: Home 10, Away 4.
    # Expected Home Pts Last 3m: 10 - 2 = 8.
    # Expected Away Pts Last 3m: 4 - 0 = 4.
    # Expected Momentum Diff: 8 - 4 = 4.
    row_700 = processed_df[processed_df['PCTIMESTRING'] == '7:00'].iloc[0]
    assert row_700['home_points_last_3_mins'] == 8, f"Expected 8, got {row_700['home_points_last_3_mins']}"
    assert row_700['away_points_last_3_mins'] == 4, f"Expected 4, got {row_700['away_points_last_3_mins']}"
    assert row_700['momentum_differential'] == 4, f"Expected 4, got {row_700['momentum_differential']}"

    print("\nTest complete. All momentum assertions passed.")
