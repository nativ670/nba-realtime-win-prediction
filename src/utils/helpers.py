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
