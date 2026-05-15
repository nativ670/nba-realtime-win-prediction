import pandas as pd
import numpy as np
import os
import sys
import re
import unicodedata
from nba_api.stats.endpoints import playbyplayv3, boxscoretraditionalv3, boxscoresummaryv3

# --- Path Injection ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

# Configuration
RAPTOR_CSV_PATH = "data/processed/current_raptor.csv"

def normalize_name(name):
    """Normalizes names by removing diacritics and converting to lowercase."""
    if not name:
        return ""
    # Normalize unicode characters (e.g., Schröder -> Schroder)
    name = unicodedata.normalize('NFD', name)
    name = "".join([c for c in name if not unicodedata.combining(c)])
    return name.lower().strip()

def load_raptor_dict():
    """
    Loads RAPTOR data from CSV and returns a dictionary mapping PLAYER_ID to RAPTOR_TOTAL.
    """
    if not os.path.exists(RAPTOR_CSV_PATH):
        print(f"Warning: {RAPTOR_CSV_PATH} not found. Using empty dictionary.")
        return {}
    
    df = pd.read_csv(RAPTOR_CSV_PATH)
    # Map ID to Total RAPTOR
    raptor_dict = df.set_index('PLAYER_ID')['RAPTOR_TOTAL'].to_dict()
    return raptor_dict

def calculate_live_floor_advantage(game_id):
    """
    Tracks live substitutions and calculates the real-time RAPTOR advantage on the floor.
    Returns the Play-by-Play DataFrame with an additional 'live_raptor_advantage' column.
    """
    # 1. Load RAPTOR data
    raptor_lookup = load_raptor_dict()
    
    # 2. Fetch Game Metadata (Home/Away Team IDs)
    summary = boxscoresummaryv3.BoxScoreSummaryV3(game_id=game_id)
    summary_df = summary.get_data_frames()[0]
    home_team_id = summary_df.iloc[0]['homeTeamId']
    away_team_id = summary_df.iloc[0]['awayTeamId']
    
    # 3. Fetch Box Score Traditional to identify starters and map names to IDs
    box = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
    box_df = box.get_data_frames()[0]
    
    # Map for name resolution: (NormalizedFamilyName, TeamID) -> PersonID
    name_team_to_id = {}
    player_id_to_name = {}
    
    home_on_floor = set()
    away_on_floor = set()
    
    for _, row in box_df.iterrows():
        pid = row['personId']
        tid = row['teamId']
        fam_name = normalize_name(row['familyName'])
        full_name = f"{row['firstName']} {row['familyName']}"
        
        name_team_to_id[(fam_name, tid)] = pid
        player_id_to_name[pid] = full_name
        
        # Identify starters
        if row['position'] and row['position'] != '':
            if tid == home_team_id:
                home_on_floor.add(pid)
            else:
                away_on_floor.add(pid)
                
    # 4. Fetch Play-by-Play Data
    pbp = playbyplayv3.PlayByPlayV3(game_id=game_id)
    pbp_df = pbp.get_data_frames()[0].copy()
    
    # Sort by actionNumber to ensure chronological processing
    pbp_df = pbp_df.sort_values('actionNumber')
    
    live_advantages = []
    
    # Regex to parse substitution description: "SUB: [In] FOR [Out]"
    sub_pattern = re.compile(r"SUB:\s+(.*?)\s+FOR\s+(.*)")

    for _, row in pbp_df.iterrows():
        action_type = row['actionType']
        desc = row['description'] if pd.notna(row['description']) else ""
        
        if action_type == 'Substitution':
            tid = row['teamId']
            out_pid = row['personId']
            
            match = sub_pattern.search(desc)
            if match:
                in_name_parsed = normalize_name(match.group(1))
                in_pid = None
                
                # Check for direct family name match
                if (in_name_parsed, tid) in name_team_to_id:
                    in_pid = name_team_to_id[(in_name_parsed, tid)]
                else:
                    # Flexible search for initials or partial matches
                    for (fam, t), pid in name_team_to_id.items():
                        if t == tid and (in_name_parsed.endswith(fam) or fam in in_name_parsed):
                            in_pid = pid
                            break
                
                if in_pid:
                    target_set = home_on_floor if tid == home_team_id else away_on_floor
                    if out_pid in target_set:
                        target_set.remove(out_pid)
                    target_set.add(in_pid)
                else:
                    print(f"Warning: Could not resolve entering player '{match.group(1)}' for team {tid}")

        # Calculate RAPTOR sums
        def get_team_raptor(player_set):
            total = 0.0
            for pid in player_set:
                total += raptor_lookup.get(pid, -1.0) # Default to -1.0 for missing players
            return total

        home_raptor = get_team_raptor(home_on_floor)
        away_raptor = get_team_raptor(away_on_floor)
        advantage = home_raptor - away_raptor
        live_advantages.append(advantage)

    pbp_df['live_raptor_advantage'] = live_advantages
    return pbp_df

if __name__ == "__main__":
    # Test on a recent game ID (CLE @ DET from 2026-05-14)
    TEST_GAME_ID = '0042500205'
    
    try:
        results_df = calculate_live_floor_advantage(TEST_GAME_ID)
        
        print("\n--- 10-MAN TRUTH: LIVE SUBSTITUTION TRACKER ---")
        # Display the first 20 rows where advantage changes or a sub happens
        # Or just the first 20 rows as requested
        cols_to_show = ['clock', 'period', 'description', 'actionType', 'live_raptor_advantage']
        
        # Filter for rows that are interesting (subs or start of period)
        # But user asked for first 20 rows.
        print(results_df[cols_to_show].head(40))
        
        # Find some subs to prove it's working
        subs = results_df[results_df['actionType'] == 'Substitution'].head(5)
        if not subs.empty:
            print("\n--- SAMPLE SUBSTITUTION EVENTS ---")
            print(subs[cols_to_show])
        else:
            print("\nNo substitution events found in the first batch.")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
