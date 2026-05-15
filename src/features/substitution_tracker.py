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

class SubstitutionTracker:
    def __init__(self, game_id, raptor_lookup=None):
        self.game_id = game_id
        self.raptor_lookup = raptor_lookup or load_raptor_dict()
        self.home_team_id = None
        self.away_team_id = None
        self.home_on_floor = set()
        self.away_on_floor = set()
        self.name_team_to_id = {}
        self.player_id_to_name = {}
        self.initialized = False
        self.sub_pattern = re.compile(r"SUB:\s+(.*?)\s+FOR\s+(.*)")

    def _initialize_metadata(self):
        """Fetches metadata and starters for the game."""
        try:
            summary = boxscoresummaryv3.BoxScoreSummaryV3(game_id=self.game_id)
            summary_df = summary.get_data_frames()[0]
            self.home_team_id = summary_df.iloc[0]['homeTeamId']
            self.away_team_id = summary_df.iloc[0]['awayTeamId']

            box = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=self.game_id)
            box_df = box.get_data_frames()[0]

            for _, row in box_df.iterrows():
                pid = row['personId']
                tid = row['teamId']
                fam_name = normalize_name(row['familyName'])
                full_name = f"{row['firstName']} {row['familyName']}"
                
                self.name_team_to_id[(fam_name, tid)] = pid
                self.player_id_to_name[pid] = full_name
                
                if row['position'] and row['position'] != '':
                    if tid == self.home_team_id:
                        self.home_on_floor.add(pid)
                    else:
                        self.away_on_floor.add(pid)
            
            self.initialized = True
        except Exception as e:
            print(f"Error initializing SubstitutionTracker: {e}")

    def process_pbp(self, pbp_df):
        """Processes a PBP DataFrame and adds the live_raptor_advantage column."""
        if not self.initialized:
            self._initialize_metadata()
        
        # Reset floor sets to starters before processing history
        # (Since PBP usually contains the full game history)
        self.home_on_floor = set()
        self.away_on_floor = set()
        self._initialize_metadata() # Re-fetch starters

        df = pbp_df.sort_values('actionNumber').copy()
        live_advantages = []

        for _, row in df.iterrows():
            action_type = row.get('actionType') or row.get('EVENTMSGTYPE_STR', '')
            desc = row['description'] if pd.notna(row['description']) else ""
            
            if action_type == 'Substitution' or row.get('EVENTMSGTYPE') == 8:
                tid = row.get('teamId') or row.get('PLAYER1_TEAM_ID')
                out_pid = row.get('personId') # Should be updated for V2 compatibility if needed
                
                match = self.sub_pattern.search(desc)
                if match:
                    in_name_parsed = normalize_name(match.group(1))
                    in_pid = None
                    
                    if (in_name_parsed, tid) in self.name_team_to_id:
                        in_pid = self.name_team_to_id[(in_name_parsed, tid)]
                    else:
                        for (fam, t), pid in self.name_team_to_id.items():
                            if t == tid and (in_name_parsed.endswith(fam) or fam in in_name_parsed):
                                in_pid = pid
                                break
                    
                    if in_pid:
                        target_set = self.home_on_floor if tid == self.home_team_id else self.away_on_floor
                        # If we don't have out_pid (e.g. from V2 PBP), we can't reliably remove.
                        # But V3 provides personId for the OUT player.
                        if out_pid in target_set:
                            target_set.remove(out_pid)
                        target_set.add(in_pid)

            def get_team_raptor(player_set):
                total = 0.0
                for pid in player_set:
                    total += self.raptor_lookup.get(pid, -1.0)
                return total

            home_raptor = get_team_raptor(self.home_on_floor)
            away_raptor = get_team_raptor(self.away_on_floor)
            live_advantages.append(home_raptor - away_raptor)

        df['live_raptor_advantage'] = live_advantages
        return df

def calculate_live_floor_advantage(game_id):
    """Legacy wrapper for the new Tracker class."""
    pbp = playbyplayv3.PlayByPlayV3(game_id=game_id)
    pbp_df = pbp.get_data_frames()[0]
    tracker = SubstitutionTracker(game_id)
    return tracker.process_pbp(pbp_df)

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
