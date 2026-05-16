import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add the project root to sys.path so 'src' can be found when running directly
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_path not in sys.path:
    sys.path.append(root_path)

# Import constants and fetcher from project structure
from src.utils.helpers import TEAM_COORDS, TEAM_ABBREV_MAP, ELO_TEAM_MAP
from src.data_ingestion.fetch_elo import fetch_elo_data

# ==============================================================================
# GEOSPATIAL UTILITY
# ==============================================================================

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on the earth
    (specified in decimal degrees) using the Haversine formula.
    Returns distance in miles. Vectorized for Pandas performance.
    """
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 3963.2 # Radius of earth in miles
    return c * r

# ==============================================================================
# FEATURE ENGINEERING LOGIC
# ==============================================================================

def calculate_pregame_features(df_raw):
    """
    Primary feature engineering engine. 
    Translates R data manipulation into vectorized Pandas operations.
    Focuses on travel, rest, and basic performance.
    """
    df = df_raw.copy()
    
    # 1. Standardize Team Abbreviations
    df['team_abbreviation'] = df['team_abbreviation'].replace(TEAM_ABBREV_MAP)
    
    # 2. Add Team Coordinates
    coords_df = pd.DataFrame.from_dict(TEAM_COORDS, orient='index', columns=['lat', 'lon']).reset_index()
    coords_df.columns = ['team_abbreviation', 'lat', 'lon']
    
    # Join for my team
    df = df.merge(coords_df, on='team_abbreviation', how='left')
    df = df.rename(columns={'lat': 'my_home_lat', 'lon': 'my_home_lon'})
    
    # Join for opponent team
    team_id_map = df[['team_id', 'team_abbreviation']].drop_duplicates().set_index('team_id')['team_abbreviation']
    df['opp_abbrev_safe'] = df['opponent_team_id'].map(team_id_map)
    
    df = df.merge(coords_df.rename(columns={'team_abbreviation': 'opp_abbrev_safe', 'lat': 'opp_home_lat', 'lon': 'opp_home_lon'}),
                  on='opp_abbrev_safe', how='left')

    # 3. Handle Venue Edge Cases (Historical Moves)
    # Sacramento Kings (Moved to Golden 1 Center in 2016-17)
    mask_sac_pre_2017 = (df['team_abbreviation'] == 'SAC') & (df['season'] < 2017)
    df.loc[mask_sac_pre_2017, ['my_home_lat', 'my_home_lon']] = [38.64, -121.51]
    
    mask_opp_sac_pre_2017 = (df['opp_abbrev_safe'] == 'SAC') & (df['season'] < 2017)
    df.loc[mask_opp_sac_pre_2017, ['opp_home_lat', 'opp_home_lon']] = [38.64, -121.51]

    # Detroit Pistons (Moved to Little Caesars Arena in 2017-18)
    mask_det_pre_2018 = (df['team_abbreviation'] == 'DET') & (df['season'] < 2018)
    df.loc[mask_det_pre_2018, ['my_home_lat', 'my_home_lon']] = [42.69, -83.24]
    
    mask_opp_det_pre_2018 = (df['opp_abbrev_safe'] == 'DET') & (df['season'] < 2018)
    df.loc[mask_opp_det_pre_2018, ['opp_home_lat', 'opp_home_lon']] = [42.69, -83.24]

    # Golden State Warriors (Moved to Chase Center in 2019-20)
    mask_gsw_pre_2020 = (df['team_abbreviation'] == 'GSW') & (df['season'] < 2020)
    df.loc[mask_gsw_pre_2020, ['my_home_lat', 'my_home_lon']] = [37.75, -122.20]
    
    mask_opp_gsw_pre_2020 = (df['opp_abbrev_safe'] == 'GSW') & (df['season'] < 2020)
    df.loc[mask_opp_gsw_pre_2020, ['opp_home_lat', 'opp_home_lon']] = [37.75, -122.20]

    # Toronto Raptors (2021 Season) moved to Amalie Arena in Tampa, FL
    mask_tor_2021 = (df['team_abbreviation'] == 'TOR') & (df['season'] == 2021)
    df.loc[mask_tor_2021, ['my_home_lat', 'my_home_lon']] = [27.94, -82.45]
    
    mask_opp_tor_2021 = (df['opp_abbrev_safe'] == 'TOR') & (df['season'] == 2021)
    df.loc[mask_opp_tor_2021, ['opp_home_lat', 'opp_home_lon']] = [27.94, -82.45]
    
    # LA Clippers (2025 Season) moved to Intuit Dome in Inglewood, CA
    mask_lac_2025 = (df['team_abbreviation'] == 'LAC') & (df['season'] >= 2025)
    df.loc[mask_lac_2025, ['my_home_lat', 'my_home_lon']] = [33.94, -118.34]
    
    mask_opp_lac_2025 = (df['opp_abbrev_safe'] == 'LAC') & (df['season'] >= 2025)
    df.loc[mask_opp_lac_2025, ['opp_home_lat', 'opp_home_lon']] = [33.94, -118.34]

    # 4. Handle Neutral Sites (Bubble & Global Games)
    # NBA Bubble (Orlando 2020)
    # Most games from July 30, 2020, to October 11, 2020, were in the Disney Bubble
    bubble_start = '2020-07-30'
    bubble_end = '2020-10-11'
    mask_bubble = df['game_date'].between(bubble_start, bubble_end)
    df.loc[mask_bubble, ['game_lat', 'game_lon']] = [28.37, -81.55]

    # Note: Global Games (Mexico, Paris, London) are rarer but can be added if game_id is known.
    # For now, we prioritize the Bubble as it impacts hundreds of games.

    # 5. Determine Game Location Coordinates (for non-bubble games)
    default_lat = pd.Series(np.where(df['team_home_away'] == 'home', df['my_home_lat'], df['opp_home_lat']), index=df.index)
    default_lon = pd.Series(np.where(df['team_home_away'] == 'home', df['my_home_lon'], df['opp_home_lon']), index=df.index)
    
    df['game_lat'] = df['game_lat'].fillna(default_lat)
    df['game_lon'] = df['game_lon'].fillna(default_lon)

    # 6. Time and Travel Features (Vectorized GroupBy)
    df = df.sort_values(['team_id', 'game_date'])
    df['game_date'] = pd.to_datetime(df['game_date'])
    
    # Rest Days
    df['prev_game_date'] = df.groupby('team_id')['game_date'].shift(1)
    df['days_rest'] = (df['game_date'] - df['prev_game_date']).dt.days - 1
    df['days_rest'] = df['days_rest'].fillna(7) # Default to 7 if no prior game
    df['days_rest_capped'] = df['days_rest'].clip(upper=5)
    
    # Back-to-Back Indicators
    df['b2b_second_game'] = (df['days_rest'] == 0).astype(int)
    
    df['next_game_date'] = df.groupby('team_id')['game_date'].shift(-1)
    df['days_until_next'] = (df['next_game_date'] - df['game_date']).dt.days - 1
    df['b2b_first_game'] = (df['days_until_next'] == 0).astype(int)

    # Distance Traveled (Haversine)
    df['prev_game_lat'] = df.groupby('team_id')['game_lat'].shift(1)
    df['prev_game_lon'] = df.groupby('team_id')['game_lon'].shift(1)
    
    df['distance_traveled'] = haversine_vectorized(
        df['prev_game_lat'], df['prev_game_lon'],
        df['game_lat'], df['game_lon']
    ).fillna(0)
    
    df['is_home'] = (df['team_home_away'] == 'home').astype(int)

    # 6. Performance Metrics (Net Rating)
    df['possessions'] = 0.96 * (
        df['field_goals_attempted'] + 
        (0.44 * df['free_throws_attempted']) - 
        df['offensive_rebounds'] + 
        df['turnovers']
    )
    df['point_diff'] = df['team_score'] - df['opponent_team_score']
    df['net_rating'] = (df['point_diff'] / df['possessions'].replace(0, np.nan)) * 100
    
    return df

def add_external_features(df, elo_df):
    """
    Merges Elo data and calculates relative features like Rest Advantage.
    Optimized for playoff awareness and using both teams' Elo ratings.
    """
    # 1. Prep Elo Data for Join
    # Neil Paine dataset uses home (team1) and away (team2)
    elo_cols = ['date', 'team1', 'team2', 'elo1_pre', 'elo2_pre', 'playoff']
    elo_clean = elo_df[elo_cols].copy()
    elo_clean['date'] = pd.to_datetime(elo_clean['date'], format='mixed')
    
    # Standardize our team abbreviations to match Neil Paine's
    df['neil_team'] = df['team_abbreviation'].replace(ELO_TEAM_MAP)
    df['neil_opp'] = df['opp_abbrev_safe'].replace(ELO_TEAM_MAP)
    
    # 2. Join Elo (Handling Home/Away context)
    # Join where our team is Team1 (Home)
    df_home = df[df['team_home_away'] == 'home'].merge(
        elo_clean, 
        left_on=['game_date', 'neil_team', 'neil_opp'],
        right_on=['date', 'team1', 'team2'],
        how='inner'
    )
    df_home['my_elo_pre'] = df_home['elo1_pre']
    df_home['opp_elo_pre'] = df_home['elo2_pre']
    
    # Join where our team is Team2 (Away)
    df_away = df[df['team_home_away'] == 'away'].merge(
        elo_clean, 
        left_on=['game_date', 'neil_team', 'neil_opp'],
        right_on=['date', 'team2', 'team1'],
        how='inner'
    )
    df_away['my_elo_pre'] = df_away['elo2_pre']
    df_away['opp_elo_pre'] = df_away['elo1_pre']
    
    # Recombine
    df = pd.concat([df_home, df_away]).sort_values(['team_id', 'game_date'])
    
    # 3. Elo-based Features
    df['elo_diff'] = df['my_elo_pre'] - df['opp_elo_pre']
    df['elo_advantage'] = df['elo_diff'] / 100 # Scaled for model stability
    
    # 4. Playoff Context Logic
    # In playoffs, rest advantage and travel impact are often mitigated by series focus
    df['is_playoffs'] = df['playoff'].fillna(0).astype(int)
    
    # 5. Rest Advantage
    # We need the opponent's days_rest_capped for the same game_id
    opp_rest_df = df[['game_id', 'team_id', 'days_rest_capped']].rename(
        columns={'team_id': 'opponent_team_id', 'days_rest_capped': 'opp_rest'}
    )
    df = df.merge(opp_rest_df, on=['game_id', 'opponent_team_id'], how='left')
    
    df['rest_advantage'] = df['days_rest_capped'] - df['opp_rest']
    
    # In playoffs, B2Bs don't exist, but "rest advantage" can come from series length
    # We'll keep the column but acknowledge it might be 0 for most playoff games.
    
    # 6. NBA Cup Indicators (In-Season Tournament - Regular Season Only)
    df['wday_num'] = df['game_date'].dt.dayofweek 
    df['is_nba_cup_group'] = 0
    
    mask_cup = (
        ((df['season'] == 2024) & (df['game_date'].between('2023-11-03', '2023-12-09'))) |
        ((df['season'] == 2025) & (df['game_date'].between('2024-11-12', '2024-12-17')))
    ) & (df['wday_num'].isin([1, 4])) & (df['is_playoffs'] == 0)
    
    df.loc[mask_cup, 'is_nba_cup_group'] = 1
    
    return df

# ==============================================================================
# MAIN EXECUTION (FOR TESTING)
# ==============================================================================

if __name__ == "__main__":
    # Lightweight functional test with mock data
    print("RUNNING LIGHTWEIGHT PREGAME TEST...")
    mock_data = pd.DataFrame({
        'team_id': [1610612738, 1610612744], # Celtics, Warriors
        'team_abbreviation': ['BOS', 'GS'],
        'opponent_team_id': [1610612744, 1610612738],
        'game_id': ['0022300001', '0022300001'],
        'game_date': ['2023-10-24', '2023-10-24'],
        'season': [2024, 2024],
        'team_home_away': ['home', 'away'],
        'team_score': [108, 104],
        'opponent_team_score': [104, 108],
        'field_goals_attempted': [88, 92],
        'free_throws_attempted': [20, 25],
        'offensive_rebounds': [10, 12],
        'turnovers': [15, 13]
    })
    
    # 1. Test basic features
    df_pre = calculate_pregame_features(mock_data)
    print(f"Features calculated: {df_pre.columns.tolist()[:10]}...")
    
    # 2. Test external features (Requires actual Elo data for realistic join)
    try:
        elo_df = fetch_elo_data()
        if not elo_df.empty:
            df_final = add_external_features(df_pre, elo_df.head(1000)) # Small subset
            print(f"Final features (with Elo): {df_final[['my_elo_pre', 'elo_advantage', 'is_playoffs']].iloc[0].to_dict()}")
    except Exception as e:
        print(f"Elo join test skipped or failed: {e}")
    
    print("Test complete.")
