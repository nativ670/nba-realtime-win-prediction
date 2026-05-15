import pandas as pd

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

def standardize_pbp_v3(df_pbp):
    """
    Standardizes PlayByPlayV3 data to the legacy V2 format 
    expected by the feature engineering engine.
    """
    if df_pbp is None or df_pbp.empty:
        return df_pbp
        
    # Mapping from PlayByPlayV3 (camelCase) to legacy V2 format
    rename_map = {
        'gameId': 'GAME_ID',
        'period': 'PERIOD',
        'clock': 'PCTIMESTRING',
        'scoreHome': 'SCORE_HOME',
        'scoreAway': 'SCORE_AWAY',
        'teamId': 'PLAYER1_TEAM_ID',
        'actionId': 'EVENTNUM',
        'actionType': 'EVENTMSGTYPE_STR'
    }
    df = df_pbp.rename(columns=rename_map)

    # Convert PCTIMESTRING from 'PT12M00.00S' to '12:00'
    def clean_clock(clock_str):
        if not clock_str or not isinstance(clock_str, str):
            return "0:00"
        if 'PT' in clock_str:
            # Extract minutes and seconds from ISO-8601 like duration
            # Example: PT11M58.00S -> 11:58
            parts = clock_str.replace('PT', '').replace('S', '').split('M')
            if len(parts) == 2:
                mins = parts[0].lstrip('0') or '0'
                secs = parts[1].split('.')[0]
                return f"{mins}:{secs.zfill(2)}"
            elif len(parts) == 1: # Only seconds?
                secs = parts[0].split('.')[0]
                return f"0:{secs.zfill(2)}"
        return clock_str

    df['PCTIMESTRING'] = df['PCTIMESTRING'].apply(clean_clock)
    
    # Reconstruct 'SCORE' column: "AWAY - HOME"
    if 'SCORE_HOME' in df.columns and 'SCORE_AWAY' in df.columns:
        df['SCORE'] = df['SCORE_AWAY'].astype(str) + " - " + df['SCORE_HOME'].astype(str)
    
    # Map actionType to EVENTMSGTYPE (Heuristic)
    # 1=Make, 2=Miss, 4=Rebound, 5=Turnover, 6=Foul, 9=Timeout
    event_map = {
        'Made Shot': 1,
        'Missed Shot': 2,
        'Rebound': 4,
        'Turnover': 5,
        'Timeout': 9,
        'Foul': 6,
        'Violation': 7,
        'Substitution': 8,
        'period': 10,
        'Jump Ball': 11,
        'Free Throw': 3
    }
    if 'EVENTMSGTYPE_STR' in df.columns:
        df['EVENTMSGTYPE'] = df['EVENTMSGTYPE_STR'].map(event_map).fillna(0)
    
    return df

if __name__ == "__main__":
    print("TEAM_COORDS sample (BOS):", TEAM_COORDS["BOS"])
    print("TEAM_ABBREV_MAP sample (GS -> GSW):", TEAM_ABBREV_MAP["GS"])
    print("ELO_TEAM_MAP sample (BKN -> BRK):", ELO_TEAM_MAP["BKN"])
    print("Helpers test complete.")
