import pandas as pd
import numpy as np
import os
from src.features.in_game import calculate_in_game_features
from src.features.pregame import calculate_pregame_features, add_external_features
from src.data_ingestion.fetch_elo import fetch_elo_data

# ==============================================================================
# PIPELINE CONFIGURATION
# ==============================================================================

RAW_PBP_PATH = "data/raw/nba_pbp_10y.parquet"
RAW_BOX_PATH = "data/raw/nba_box_scores_10y.parquet"
PROCESSED_DATA_PATH = "data/processed/training_data.parquet"

# ==============================================================================
# DATASET BUILDER ENGINE
# ==============================================================================

def build_training_dataset():
    """
    Master pipeline to merge PBP, In-Game features, Pre-Game features, and Target.
    """
    print(f"Loading raw data...")
    if not os.path.exists(RAW_PBP_PATH):
        raise FileNotFoundError(f"PBP data not found at {RAW_PBP_PATH}")
    if not os.path.exists(RAW_BOX_PATH):
        raise FileNotFoundError(f"Box score data not found at {RAW_BOX_PATH}")

    df_pbp_raw = pd.read_parquet(RAW_PBP_PATH)
    df_box_raw = pd.read_parquet(RAW_BOX_PATH)

    # 1. STANDARDIZE PBP COLUMNS
    # The raw PBP from nba_api has camelCase; we convert to the expected format for in_game.py
    df_pbp = df_pbp_raw.rename(columns={
        'gameId': 'GAME_ID',
        'period': 'PERIOD',
        'clock': 'PCTIMESTRING',
        'scoreHome': 'SCORE_HOME',
        'scoreAway': 'SCORE_AWAY',
        'teamId': 'PLAYER1_TEAM_ID',
        'actionId': 'EVENTNUM' # Approximate mapping for mock logic
    })

    # Convert PCTIMESTRING from 'PT12M00.00S' to '12:00'
    def clean_clock(clock_str):
        if not clock_str or not isinstance(clock_str, str):
            return "0:00"
        # Extract minutes and seconds from ISO-8601 like duration
        # Example: PT11M58.00S -> 11:58
        parts = clock_str.replace('PT', '').replace('S', '').split('M')
        mins = parts[0]
        secs = parts[1].split('.')[0]
        return f"{mins}:{secs.zfill(2)}"

    df_pbp['PCTIMESTRING'] = df_pbp['PCTIMESTRING'].apply(clean_clock)
    
    # Reconstruct 'SCORE' column for in_game.py: "AWAY - HOME" (based on in_game.py logic)
    df_pbp['SCORE'] = df_pbp['SCORE_AWAY'].astype(str) + " - " + df_pbp['SCORE_HOME'].astype(str)
    
    # Map actionType/subType to EVENTMSGTYPE (Heuristic)
    # 1=Make, 2=Miss, 4=Rebound, 5=Turnover
    event_map = {
        'made': 1,
        'missed': 2,
        'rebound': 4,
        'turnover': 5
    }
    df_pbp['EVENTMSGTYPE'] = df_pbp['actionType'].map(event_map).fillna(0)

    # 2. CALCULATE IN-GAME FEATURES (Vectorized)
    print("Engineering in-game features...")
    # Apply calculate_in_game_features per game
    # We use group_keys=True to ensure GAME_ID is part of the index, then reset it
    df_pbp_features = df_pbp.groupby('GAME_ID', group_keys=True).apply(calculate_in_game_features)
    df_pbp_features = df_pbp_features.reset_index(level=0) # Pull GAME_ID from index back to column
    df_pbp_features = df_pbp_features.reset_index(drop=True) # Clean up the rest of the index

    # 3. DETERMINE TARGET VARIABLE (home_win)
    print("Calculating final game outcomes...")
    # Get the last row of each game to see the final score
    # We use a copy to avoid SettingWithCopyWarning
    final_scores = df_pbp.sort_values(['GAME_ID', 'EVENTNUM']).groupby('GAME_ID').tail(1).copy()
    
    # Ensure numeric types for scores
    final_scores['home_final'] = pd.to_numeric(final_scores['SCORE_HOME'], errors='coerce').fillna(0)
    final_scores['away_final'] = pd.to_numeric(final_scores['SCORE_AWAY'], errors='coerce').fillna(0)
    
    final_scores['home_win'] = (final_scores['home_final'] > final_scores['away_final']).astype(int)
    
    # Merge target back to PBP
    df_pbp_features = df_pbp_features.merge(final_scores[['GAME_ID', 'home_win']], on='GAME_ID', how='left')

    # 4. PREPARE PRE-GAME FEATURES
    print("Engineering pre-game features...")
    # Prepare box score data for pregame.py
    # We need to reshape the two rows per game into a single row with team and opponent info
    # or handle it as pregame.py expects (team-game rows).
    
    # Standardize column names for pregame.py
    df_box = df_box_raw.copy()
    
    # Pre-game features logic: needs team and opponent on the same row or grouped
    # Let's create the paired matchup for pregame processing
    df_box_paired = []
    for gid, group in df_box.groupby('game_id'):
        if len(group) == 2:
            row1 = group.iloc[0].to_dict()
            row2 = group.iloc[1].to_dict()
            
            # Row for Team 1
            r1 = row1.copy()
            r1['opponent_team_id'] = row2['team_id']
            r1['opponent_team_score'] = row2['pts']
            r1['team_home_away'] = 'home' if 'vs.' in r1['matchup'] else 'away'
            df_box_paired.append(r1)
            
            # Row for Team 2
            r2 = row2.copy()
            r2['opponent_team_id'] = row1['team_id']
            r2['opponent_team_score'] = row1['pts']
            r2['team_home_away'] = 'home' if 'vs.' in r2['matchup'] else 'away'
            df_box_paired.append(r2)
            
    df_box_pre = pd.DataFrame(df_box_paired)
    
    # Map needed columns
    df_box_pre = df_box_pre.rename(columns={
        'fga': 'field_goals_attempted',
        'fta': 'free_throws_attempted',
        'oreb': 'offensive_rebounds',
        'tov': 'turnovers',
        'pts': 'team_score'
    })

    # Run pregame feature engineering
    df_pre_features = calculate_pregame_features(df_box_pre)
    
    # Add Elo
    elo_df = fetch_elo_data()
    df_pre_features = add_external_features(df_pre_features, elo_df)

    # 5. MERGE PRE-GAME FEATURES INTO PBP
    print("Merging pre-game and in-game features...")
    # We only need one side of the pre-game features (home team perspective usually works best for WP)
    df_pre_home = df_pre_features[df_pre_features['team_home_away'] == 'home'].copy()
    
    # Select key pre-game features to merge
    pre_cols = ['game_id', 'my_elo_pre', 'opp_elo_pre', 'elo_advantage', 'days_rest_capped', 'rest_advantage', 'distance_traveled', 'is_playoffs']
    
    df_pbp_features = df_pbp_features.merge(
        df_pre_home[pre_cols].rename(columns={'game_id': 'GAME_ID'}), 
        on='GAME_ID', 
        how='left'
    )

    # 6. CLEANUP & SAVE
    print(f"Cleaning data and saving to {PROCESSED_DATA_PATH}...")
    # Drop rows without critical features (e.g., Elo)
    df_final = df_pbp_features.dropna(subset=['my_elo_pre', 'score_differential'])
    
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    df_final.to_parquet(PROCESSED_DATA_PATH, index=False)
    
    print(f"Pipeline complete! Final shape: {df_final.shape}")
    return df_final

# ==============================================================================
# MAIN EXECUTION (LIGHTWEIGHT TEST)
# ==============================================================================

if __name__ == "__main__":
    print("RUNNING PIPELINE TEST...")
    
    # Check if we have data
    if not os.path.exists(RAW_PBP_PATH):
        print(f"Error: Raw PBP file not found at {RAW_PBP_PATH}")
    else:
        # For testing, we just process a subset
        full_df = build_training_dataset()
        
        print("\nPipeline Output Summary:")
        print(f"Columns: {full_df.columns.tolist()}")
        print(f"Home Win Distribution:\n{full_df['home_win'].value_counts(normalize=True)}")
        
        # Verify a few rows
        sample_cols = [
            'GAME_ID', 'seconds_remaining_in_game', 'score_differential', 
            'home_points_last_3_mins', 'away_points_last_3_mins', 'momentum_differential',
            'possession_team_id', 'elo_advantage', 'home_win'
        ]
        print("\nSample Rows:")
        print(full_df[sample_cols].head())
