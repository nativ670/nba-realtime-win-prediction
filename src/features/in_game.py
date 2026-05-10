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
    """
    df = df_pbp.copy()

    # 1. TIME REMAINING CALCULATION
    # Convert 'PERIOD' and 'PCTIMESTRING' to total seconds remaining in game
    # NBA Regulation: 4 periods of 12 mins (720s). Overtime: 5 mins (300s).
    
    def parse_clock(clock_str):
        if pd.isna(clock_str) or clock_str == "":
            return 0
        minutes, seconds = map(int, clock_str.split(':'))
        return (minutes * 60) + seconds

    # Vectorized clock parsing (using Series.str.split)
    clock_parts = df['PCTIMESTRING'].str.split(':', expand=True).astype(float)
    df['seconds_remaining_in_period'] = (clock_parts[0] * 60) + clock_parts[1]

    # Calculate seconds at the start of each period
    # Period 1: 2880s (48m), Period 2: 2160s (36m), Period 3: 1440s (24m), Period 4: 720s (12m)
    # OT periods: 5 mins each
    df['seconds_remaining_in_game'] = np.where(
        df['PERIOD'] <= 4,
        (4 - df['PERIOD']) * 720 + df['seconds_remaining_in_period'],
        # Handle OT: We don't know total OTs in advance, so we treat each as 300s remaining from its end
        # This is a simplification; for WP models, we usually focus on time left in current OT.
        0 + df['seconds_remaining_in_period'] 
    )

    # 2. SCORE DIFFERENTIAL
    # Raw PBP usually has 'SCORE' as "80 - 75" or similar, and only on scoring events.
    # We need to split, handle NaNs, and forward-fill.
    
    # Extract scores, handling rows with no score update (NaN)
    scores = df['SCORE'].str.split(' - ', expand=True)
    df['home_score'] = pd.to_numeric(scores[1], errors='coerce')
    df['away_score'] = pd.to_numeric(scores[0], errors='coerce')

    # Forward fill to ensure every row has the current score
    df[['home_score', 'away_score']] = df[['home_score', 'away_score']].ffill().fillna(0)
    
    df['score_differential'] = df['home_score'] - df['away_score']

    # 3. POSSESSION TRACKING (Heuristic-based)
    # We infer possession based on the last 'PLAYER1_TEAM_ID' involved in a 
    # possession-defining event (Rebound, Turnover, Made Shot).
    
    # Identify events that define possession
    # EVENTMSGTYPE: 1=Make, 2=Miss, 3=FT, 4=Rebound, 5=Turnover, 6=Foul...
    possession_events = [1, 2, 4, 5] 
    
    df['possession_team_id'] = np.nan
    mask = df['EVENTMSGTYPE'].isin(possession_events)
    df.loc[mask, 'possession_team_id'] = df.loc[mask, 'PLAYER1_TEAM_ID']
    
    # Forward fill possession
    df['possession_team_id'] = df['possession_team_id'].ffill()

    return df

# ==============================================================================
# MAIN EXECUTION (LIGHTWEIGHT TEST)
# ==============================================================================

if __name__ == "__main__":
    print("RUNNING LIGHTWEIGHT IN-GAME FEATURE TEST...")
    
    # Mock Play-by-Play Data
    # 1610612738 = Celtics (Away), 1610612744 = Warriors (Home)
    mock_pbp = pd.DataFrame({
        'EVENTNUM': [1, 2, 3, 4, 5],
        'PERIOD': [1, 1, 1, 1, 1],
        'PCTIMESTRING': ['12:00', '11:45', '11:30', '11:15', '11:00'],
        'EVENTMSGTYPE': [10, 1, 4, 2, 5], # Start, Make, Rebound, Miss, Turnover
        'SCORE': [None, '2 - 0', None, None, None],
        'PLAYER1_TEAM_ID': [None, 1610612738, 1610612738, 1610612744, 1610612744],
        'HOMEDESCRIPTION': [None, None, None, 'Missed Shot', 'Turnover'],
        'VISITORDESCRIPTION': [None, 'Made Layup', 'Defensive Rebound', None, None]
    })

    processed_df = calculate_in_game_features(mock_pbp)
    
    print("\nProcessed Features Sample:")
    cols_to_show = ['PERIOD', 'PCTIMESTRING', 'seconds_remaining_in_game', 'home_score', 'away_score', 'score_differential', 'possession_team_id']
    print(processed_df[cols_to_show])
    
    # Basic Validations
    assert processed_df.iloc[1]['away_score'] == 2, "Score parsing failed"
    assert processed_df.iloc[2]['away_score'] == 2, "Score forward-fill failed"
    assert processed_df.iloc[0]['seconds_remaining_in_game'] == 2880, "Time calculation failed"
    
    print("\nTest complete. All assertions passed.")
