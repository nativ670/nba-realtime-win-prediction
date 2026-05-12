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
    
    NBA Rule Adherence:
    - Regulation Bonus: 5th team foul OR 2nd in L2M.
    - Overtime Bonus: 4th team foul OR 2nd in L2M.
    - Timeouts: 7 for regulation (pool), 2 per OT period (no carryover).
    """
    df = df_pbp.copy()

    # 1. TIME CALCULATIONS
    clock_parts = df['PCTIMESTRING'].str.split(':', expand=True).astype(float)
    df['seconds_remaining_in_period'] = (clock_parts[0] * 60) + clock_parts[1]

    df['seconds_remaining_in_game'] = np.where(
        df['PERIOD'] <= 4,
        (4 - df['PERIOD']) * 720 + df['seconds_remaining_in_period'],
        df['seconds_remaining_in_period'] 
    )

    df['elapsed_time'] = np.where(
        df['PERIOD'] <= 4,
        (df['PERIOD'] - 1) * 720 + (720 - df['seconds_remaining_in_period']),
        2880 + (df['PERIOD'] - 5) * 300 + (300 - df['seconds_remaining_in_period'])
    )

    # 2. SCORE DIFFERENTIAL
    if 'SCORE' in df.columns and df['SCORE'].notna().any():
        scores = df['SCORE'].str.split(' - ', expand=True)
        # Ensure we have two columns after split
        if scores.shape[1] == 2:
            df['home_score'] = pd.to_numeric(scores[1], errors='coerce')
            df['away_score'] = pd.to_numeric(scores[0], errors='coerce')
        else:
            df['home_score'] = 0
            df['away_score'] = 0
    else:
        df['home_score'] = 0
        df['away_score'] = 0

    df[['home_score', 'away_score']] = df[['home_score', 'away_score']].ffill().fillna(0)
    df['score_differential'] = df['home_score'] - df['away_score']

    # 3. IDENTIFY HOME/AWAY TEAM IDs
    h_score_diff = df['home_score'].diff().fillna(0)
    a_score_diff = df['away_score'].diff().fillna(0)
    mask_h_scoring = (h_score_diff > 0) & (df['PLAYER1_TEAM_ID'].notna())
    mask_a_scoring = (a_score_diff > 0) & (df['PLAYER1_TEAM_ID'].notna())
    
    home_team_id = df.loc[mask_h_scoring, 'PLAYER1_TEAM_ID'].iloc[0] if mask_h_scoring.any() else None
    away_team_id = df.loc[mask_a_scoring, 'PLAYER1_TEAM_ID'].iloc[0] if mask_a_scoring.any() else None
    
    team_ids = [tid for tid in df['PLAYER1_TEAM_ID'].unique() if pd.notna(tid) and tid != 0]
    if home_team_id is None and len(team_ids) >= 1: home_team_id = team_ids[0]
    if away_team_id is None and len(team_ids) >= 2: away_team_id = team_ids[1]

    # 4. POSSESSION TRACKING
    df['possession_team_id'] = np.nan
    if len(team_ids) >= 2:
        t1, t2 = team_ids[0], team_ids[1]
        other_team_map = {t1: t2, t2: t1}
        mask_reb = df['EVENTMSGTYPE'] == 4
        df.loc[mask_reb, 'possession_team_id'] = df.loc[mask_reb, 'PLAYER1_TEAM_ID']
        mask_flip = df['EVENTMSGTYPE'].isin([1, 5])
        df.loc[mask_flip, 'possession_team_id'] = df.loc[mask_flip, 'PLAYER1_TEAM_ID'].map(other_team_map)
    df['possession_team_id'] = df['possession_team_id'].ffill()

    # 5. CLUTCH CONTEXT (Timeouts & Fouls)
    group_cols = ['GAME_ID', 'PERIOD'] if 'GAME_ID' in df.columns else ['PERIOD']
    
    # --- TIMEOUTS ---
    df['is_home_to'] = ((df['EVENTMSGTYPE'] == 9) & (df['PLAYER1_TEAM_ID'] == home_team_id)).astype(int)
    df['is_away_to'] = ((df['EVENTMSGTYPE'] == 9) & (df['PLAYER1_TEAM_ID'] == away_team_id)).astype(int)
    
    # Regulation Pool (P1-P4)
    reg_mask = df['PERIOD'] <= 4
    df['home_to_reg_cumsum'] = df.loc[reg_mask, 'is_home_to'].cumsum()
    df['away_to_reg_cumsum'] = df.loc[reg_mask, 'is_away_to'].cumsum()
    
    # OT Pool (P5+, Reset per period)
    df['home_to_ot_cumsum'] = df.groupby(group_cols)['is_home_to'].cumsum()
    df['away_to_ot_cumsum'] = df.groupby(group_cols)['is_away_to'].cumsum()
    
    df['home_timeouts_remaining'] = np.where(
        df['PERIOD'] <= 4,
        (7 - df['home_to_reg_cumsum']).ffill(),
        (2 - df['home_to_ot_cumsum'])
    ).clip(0)
    
    df['away_timeouts_remaining'] = np.where(
        df['PERIOD'] <= 4,
        (7 - df['away_to_reg_cumsum']).ffill(),
        (2 - df['away_to_ot_cumsum'])
    ).clip(0)
    
    # --- FOULS & BONUS ---
    df['is_h_foul'] = ((df['EVENTMSGTYPE'] == 6) & (df['PLAYER1_TEAM_ID'] == home_team_id)).astype(int)
    df['is_a_foul'] = ((df['EVENTMSGTYPE'] == 6) & (df['PLAYER1_TEAM_ID'] == away_team_id)).astype(int)
    
    df['home_team_fouls'] = df.groupby(group_cols)['is_h_foul'].cumsum()
    df['away_team_fouls'] = df.groupby(group_cols)['is_a_foul'].cumsum()
    
    # L2M rule (Same for Reg and OT)
    df['is_h_l2m_foul'] = (df['is_h_foul'] == 1) & (df['seconds_remaining_in_period'] <= 120)
    df['is_a_l2m_foul'] = (df['is_a_foul'] == 1) & (df['seconds_remaining_in_period'] <= 120)
    df['home_l2m_fouls'] = df.groupby(group_cols)['is_h_l2m_foul'].cumsum()
    df['away_l2m_fouls'] = df.groupby(group_cols)['is_a_l2m_foul'].cumsum()
    
    # Bonus Threshold: 5 in Regulation, 4 in OT
    df['foul_threshold'] = np.where(df['PERIOD'] <= 4, 5, 4)
    
    df['away_in_bonus'] = ((df['home_team_fouls'] >= df['foul_threshold']) | (df['home_l2m_fouls'] >= 2)).astype(int)
    df['home_in_bonus'] = ((df['away_team_fouls'] >= df['foul_threshold']) | (df['away_l2m_fouls'] >= 2)).astype(int)

    # 6. MOMENTUM FEATURES (Lookback 180s)
    df = df.sort_values('elapsed_time')
    df['time_lookback'] = df['elapsed_time'] - 180
    df['temp_idx'] = range(len(df))
    
    group_col = 'GAME_ID' if 'GAME_ID' in df.columns else None
    df_momentum = pd.merge_asof(
        df,
        df[['elapsed_time', 'home_score', 'away_score'] + ([group_col] if group_col else [])],
        left_on='time_lookback',
        right_on='elapsed_time',
        by=group_col,
        direction='backward',
        suffixes=('', '_hist')
    )
    
    df_momentum['home_score_hist'] = df_momentum['home_score_hist'].fillna(0)
    df_momentum['away_score_hist'] = df_momentum['away_score_hist'].fillna(0)
    df_momentum['home_points_last_3_mins'] = df_momentum['home_score'] - df_momentum['home_score_hist']
    df_momentum['away_points_last_3_mins'] = df_momentum['away_score'] - df_momentum['away_score_hist']
    df_momentum['momentum_differential'] = df_momentum['home_points_last_3_mins'] - df_momentum['away_points_last_3_mins']
    
    # Clean up
    cols_to_drop = [
        'elapsed_time', 'time_lookback', 'temp_idx', 'elapsed_time_hist', 
        'home_score_hist', 'away_score_hist', 'is_home_to', 'is_away_to',
        'home_to_reg_cumsum', 'away_to_reg_cumsum', 'home_to_ot_cumsum', 'away_to_ot_cumsum',
        'is_h_foul', 'is_a_foul', 'is_h_l2m_foul', 'home_l2m_fouls', 'away_l2m_fouls', 
        'foul_threshold', 'is_a_l2m_foul'
    ]
    # Filter only existing columns to avoid errors on drop
    actual_cols_to_drop = [c for c in cols_to_drop if c in df_momentum.columns]
    df = df_momentum.sort_values('temp_idx').drop(columns=actual_cols_to_drop)

    return df

# ==============================================================================
# MAIN EXECUTION (LIGHTWEIGHT TEST)
# ==============================================================================

if __name__ == "__main__":
    print("RUNNING NBA RULE VERIFICATION (REG vs OT)...")
    
    # Home: 1001, Away: 2002
    mock_pbp = pd.DataFrame({
        'EVENTNUM': range(1, 13),
        'PERIOD': [4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5],
        'PCTIMESTRING': [
            '2:01', '1:59', '1:30', '1:00', '0:30', # P4 (Reg)
            '5:00', '4:30', '4:00', '3:30', '3:00', '2:00', '1:00' # P5 (OT)
        ],
        'EVENTMSGTYPE': [
            6, 6, 6, 9, 6, # P4: Foul 1, Foul 2 (L2M1), Foul 3 (L2M2-Bonus), TO 1, Foul 4
            10, 6, 6, 6, 6, 9, 6 # P5: Start, Foul 1, 2, 3, 4 (Bonus), TO 1, Foul 5 (Bonus)
        ],
        'SCORE': ['0 - 0']*12,
        'PLAYER1_TEAM_ID': [1001]*12
    })

    processed_df = calculate_in_game_features(mock_pbp)
    
    print("\nProcessed Features Sample (OT Focus):")
    cols_to_show = [
        'PERIOD', 'PCTIMESTRING', 'home_timeouts_remaining', 
        'home_team_fouls', 'away_in_bonus'
    ]
    print(processed_df[cols_to_show])
    
    # Regulation Bonus (P4)
    # 3rd Home Foul at 1:30 is 2nd in L2M -> AWAY BONUS
    assert processed_df.iloc[2]['away_in_bonus'] == 1, "Reg L2M bonus failed"
    
    # OT Bonus (P5)
    # Home Team Fouls reset to 0 in OT
    assert processed_df.iloc[5]['home_team_fouls'] == 0, "OT Foul reset failed"
    # OT Timeouts reset to 2
    assert processed_df.iloc[5]['home_timeouts_remaining'] == 2, "OT Timeout reset failed"
    # OT Bonus triggered on 4th foul (Index 9 in this mock)
    assert processed_df.iloc[9]['home_team_fouls'] == 4, f"Expected 4 fouls, got {processed_df.iloc[9]['home_team_fouls']}"
    assert processed_df.iloc[9]['away_in_bonus'] == 1, "OT Bonus on 4th foul failed"

    print("\nTest complete. NBA Regulation and Overtime rules verified.")
