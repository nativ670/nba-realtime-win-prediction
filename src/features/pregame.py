import pandas as pd
import numpy as np
import io
import requests
from datetime import datetime

# ==============================================================================
# CONSTANTS & MAPPINGS
# ==============================================================================

# Specific dictionary mapping for all 30 NBA team venue coordinates (lat, lon)
TEAM_COORDS = {
    "ATL": (33.75, -84.39), "BOS": (42.36, -71.06), "BKN": (40.68, -73.97),
    "CHA": (35.22, -80.84), "CHI": (41.88, -87.62), "CLE": (41.49, -81.69),
    "DAL": (32.79, -96.81), "DEN": (39.75, -105.00), "DET": (42.34, -83.05),
    "GSW": (37.76, -122.38), "HOU": (29.75, -95.36), "IND": (39.76, -86.15),
    "LAC": (34.04, -118.26), "LAL": (34.04, -118.26), "MEM": (35.13, -90.05),
    "MIA": (25.78, -80.19), "MIL": (43.04, -87.91), "MIN": (44.97, -93.27),
    "NOP": (29.94, -90.08), "NYK": (40.75, -73.99), "OKC": (35.46, -97.51),
    "ORL": (28.53, -81.38), "PHI": (39.90, -75.17), "PHX": (33.44, -112.07),
    "POR": (45.53, -122.66), "SAC": (38.58, -121.49), "SAS": (29.42, -98.49),
    "TOR": (43.64, -79.37), "UTA": (40.76, -111.90), "WAS": (38.89, -77.02)
}

# Team Abbreviation Standardization (matches R script's case_when)
TEAM_ABBREV_MAP = {
    "GS": "GSW",
    "NO": "NOP",
    "NY": "NYK",
    "SA": "SAS",
    "UTAH": "UTA",
    "WSH": "WAS"
}

# Elo Team Mapping (for Neil Paine dataset)
ELO_TEAM_MAP = {
    "BKN": "BRK",
    "CHA": "CHO",
    "PHX": "PHO"
}

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """
    Calculate the great circle distance between two points on the earth
    (specified in decimal degrees) using the Haversine formula.
    Returns distance in miles.
    """
    # Convert decimal degrees to radians 
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    # Haversine formula 
    dlat = lat2 - lat1 
    dlon = lon2 - lon1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    
    # Radius of earth in miles (matching R's geosphere approximately)
    # R uses equatorial radius 6378137m / 1609.34m per mile approx 3963.2
    r = 3963.2 
    return c * r

def fetch_elo_data():
    """
    Downloads Neil Paine's Elo Data from GitHub.
    """
    url = "https://raw.githubusercontent.com/Neil-Paine-1/NBA-elo/main/nba_elo.csv"
    try:
        print(f"Downloading Elo data from {url}...")
        response = requests.get(url)
        response.raise_for_status()
        elo_df = pd.read_csv(io.StringIO(response.text))
        return elo_df
    except Exception as e:
        print(f"Error fetching Elo data: {e}")
        return pd.DataFrame()

# ==============================================================================
# FEATURE ENGINEERING LOGIC
# ==============================================================================

def calculate_pregame_features(df_raw):
    """
    Primary feature engineering engine. 
    Translates R data manipulation into vectorized Pandas operations.
    """
    # Create a copy to avoid SettingWithCopyWarning
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
    # First get unique team_id -> team_abbreviation mapping from current data if possible,
    # or just use the coords_df join on opponent_team_abbreviation if available.
    # The R script assumes it can find the opponent abbreviation.
    # We'll create a mapping from team_id to abbreviation to be safe.
    team_id_map = df[['team_id', 'team_abbreviation']].drop_duplicates().set_index('team_id')['team_abbreviation']
    df['opp_abbrev_safe'] = df['opponent_team_id'].map(team_id_map)
    
    # Join coordinates for opponent
    df = df.merge(coords_df.rename(columns={'team_abbreviation': 'opp_abbrev_safe', 'lat': 'opp_home_lat', 'lon': 'opp_home_lon'}),
                  on='opp_abbrev_safe', how='left')

    # 3. Handle Venue Edge Cases (Toronto 2021, Clippers 2025)
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

    # 4. Determine Game Location Coordinates
    df['game_lat'] = np.where(df['team_home_away'] == 'home', df['my_home_lat'], df['opp_home_lat'])
    df['game_lon'] = np.where(df['team_home_away'] == 'home', df['my_home_lon'], df['opp_home_lon'])

    # 5. Time and Travel Features (Vectorized GroupBy)
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
    
    # Distance Category (Matches R's cut)
    df['dist_cat'] = pd.cut(df['distance_traveled'], 
                            bins=[-1, 200, 1500, 100000], 
                            labels=['Short (<200)', 'Medium', 'Long (>1500)'])
    
    df['is_home'] = (df['team_home_away'] == 'home').astype(int)

    # 6. Performance Metrics (Net Rating)
    # Formula: (Point Diff / Possessions) * 100
    # Possessions Estimate: 0.96 * (FGA + 0.44 * FTA - ORB + TOV)
    df['possessions'] = 0.96 * (
        df['field_goals_attempted'] + 
        (0.44 * df['free_throws_attempted']) - 
        df['offensive_rebounds'] + 
        df['turnovers']
    )
    df['point_diff'] = df['team_score'] - df['opponent_team_score']
    df['net_rating'] = (df['point_diff'] / df['possessions']) * 100
    
    return df

def add_external_features(df, elo_df):
    """
    Merges Elo data and calculates relative features like Rest Advantage.
    """
    # Neil Paine Team Mapping
    df['neil_team_abbrev'] = df['team_abbreviation'].replace(ELO_TEAM_MAP)
    
    # Merge Elo
    # Note: elo_df columns are usually [date, team1, elo1_pre, team2, elo2_pre, ...]
    # The R scriptrenames elo2_pre but joins on team1. 
    # Let's align with the R script's join logic: 
    # inner_join(elo_data, by = c("game_date", "neil_team_abbrev" = "team1"))
    elo_subset = elo_df[['date', 'team1', 'elo2_pre']].copy()
    elo_subset['date'] = pd.to_datetime(elo_subset['date'])
    
    df = df.merge(elo_subset, 
                  left_on=['game_date', 'neil_team_abbrev'], 
                  right_on=['date', 'team1'], 
                  how='inner')
    
    # 7. Rest Advantage
    # We need the opponent's days_rest_capped for the same game_id
    opp_rest_df = df[['game_id', 'team_id', 'days_rest_capped']].rename(
        columns={'team_id': 'opponent_team_id', 'days_rest_capped': 'opp_rest'}
    )
    df = df.merge(opp_rest_df, on=['game_id', 'opponent_team_id'], how='left')
    
    df['rest_advantage'] = df['days_rest_capped'] - df['opp_rest']
    df['elo_opp_scaled'] = df['elo2_pre'] / 100
    
    # 8. NBA Cup Indicators (In-Season Tournament)
    df['month'] = df['game_date'].dt.month
    df['wday_num'] = df['game_date'].dt.dayofweek # Mon=0, Fri=4, Tue=1
    
    df['is_nba_cup_group'] = 0
    
    # 2024 Cup (Nov 3 - Dec 9, Tuesdays (1) & Fridays (4))
    mask_2024 = (
        (df['season'] == 2024) & 
        (df['game_date'] >= '2023-11-03') & 
        (df['game_date'] <= '2023-12-09') & 
        (df['wday_num'].isin([1, 4]))
    )
    df.loc[mask_2024, 'is_nba_cup_group'] = 1
    
    # 2025 Cup (Nov 12 - Dec 17, Tuesdays (1) & Fridays (4))
    mask_2025 = (
        (df['season'] == 2025) & 
        (df['game_date'] >= '2024-11-12') & 
        (df['game_date'] <= '2024-12-17') & 
        (df['wday_num'].isin([1, 4]))
    )
    df.loc[mask_2025, 'is_nba_cup_group'] = 1
    
    return df

# ==============================================================================
# MAIN EXECUTION (FOR TESTING)
# ==============================================================================

if __name__ == "__main__":
    # Example usage:
    # df_raw = pd.read_csv("data/raw/nba_box_scores.csv")
    # elo_df = fetch_elo_data()
    # df_processed = calculate_pregame_features(df_raw)
    # df_final = add_external_features(df_processed, elo_df)
    print("Pregame Feature Engineering script loaded.")
