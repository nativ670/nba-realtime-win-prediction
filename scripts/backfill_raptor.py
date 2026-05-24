import pandas as pd
import numpy as np
import os

import glob
import re
import unicodedata
from tqdm import tqdm
import time
from nba_api.stats.endpoints import leaguedashplayerstats
from src.utils.nba_client import nba_api_call

from src.config import RAW_DIR, PROCESSED_DIR, SEASONS_DIR

# Paths now imported from config

def normalize_name(name):
    """Normalizes names by removing diacritics and converting to lowercase."""
    if not name or not isinstance(name, str):
        return ""
    name = unicodedata.normalize('NFD', name)
    name = "".join([c for c in name if not unicodedata.combining(c)])
    return name.lower().strip()

def load_master_raptor_dictionary():
    """
    Stitches three eras of RAPTOR data together.
    Returns a dictionary mapping (normalized_name, season) -> raptor_total.
    """
    master_dict = {}
    print("Loading Master RAPTOR Dictionary...")
    
    # 1. Source 1: historical_RAPTOR_by_player.csv
    hist_path = os.path.join(RAW_DIR, "historical_RAPTOR_by_player.csv")
    if os.path.exists(hist_path):
        print(f"  - Loading historical data from {hist_path}")
        df_hist = pd.read_csv(hist_path)
        # Ignore 2023 data in this file as we replace it with our complete proxy
        df_hist = df_hist[df_hist['season'] < 2023]
        for _, row in df_hist.iterrows():
            name = normalize_name(row['player_name'])
            master_dict[(name, int(row['season']))] = float(row['raptor_total'])
    
    # 2. Source 2: Proxy files for 2023, 2024, 2025
    for year in [2023, 2024, 2025]:
        proxy_path = os.path.join(RAW_DIR, f"proxy_raptor_{year}.csv")
        if os.path.exists(proxy_path):
            print(f"  - Loading proxy data from {proxy_path}")
            df_proxy = pd.read_csv(proxy_path)
            for _, row in df_proxy.iterrows():
                name = normalize_name(row['player_name'])
                master_dict[(name, year)] = float(row['raptor_total'])
                
    # 3. Source 3: current_raptor.csv (mapped to 2026)
    curr_path = os.path.join(PROCESSED_DIR, "current_raptor.csv")
    if os.path.exists(curr_path):
        print(f"  - Loading current data from {curr_path}")
        df_curr = pd.read_csv(curr_path)
        for _, row in df_curr.iterrows():
            name = normalize_name(row['PLAYER_NAME'])
            master_dict[(name, 2026)] = float(row['RAPTOR_TOTAL'])
            
    print(f"Master dictionary loaded with {len(master_dict)} entries.")
    return master_dict

def get_season_player_map(season_str):
    """Fetches all players for a season to help resolve PBP names to full names."""
    print(f"Fetching player resolution map for {season_str}...")
    try:
        stats = nba_api_call(
            leaguedashplayerstats.LeagueDashPlayerStats,
            df_index=0,
            per_mode_detailed='PerGame',
            season=season_str,
            season_type_all_star='Regular Season'
        )
        
        # Map (Normalized Family Name, TeamID) -> Full Name
        # This helps resolve PBP "Horford" to "Al Horford"
        res_map = {}
        for _, row in stats.iterrows():
            full_name = row['PLAYER_NAME']
            tid = int(row['TEAM_ID'])
            
            # Add full name resolution
            res_map[(normalize_name(full_name), tid)] = full_name
            
            # Add family name resolution
            parts = full_name.split(' ')
            fam_name = normalize_name(parts[-1])
            res_map[(fam_name, tid)] = full_name
            
            # Handle suffixes like "Jr.", "III"
            if len(parts) > 2 and parts[-1] in ["Jr.", "III", "II", "IV", "Sr."]:
                fam_name_with_suffix = normalize_name(parts[-2]) # e.g. "Nance" in "Larry Nance Jr."
                res_map[(fam_name_with_suffix, tid)] = full_name
                
        return res_map
    except Exception as e:
        print(f"Error fetching resolution map: {e}")
        return {}

def backfill_raptor_advantage():
    """
    Main pipeline to backfill live_raptor_advantage across all seasonal parquet files.
    """
    master_raptor = load_master_raptor_dictionary()
    files = sorted(glob.glob(os.path.join(SEASONS_DIR, "pbp_*.parquet")))
    
    if not files:
        print("No seasonal parquet files found.")
        return

    # Cache for player resolution maps to avoid repeated API calls
    resolution_cache = {}

    sub_pattern = re.compile(r"SUB:\s+(.*?)\s+FOR\s+(.*)")

    for file_path in files:
        # Extract season year from filename (e.g., pbp_2024.parquet)
        match = re.search(r"pbp_(\d+).parquet", file_path)
        if not match: continue
        parquet_season = int(match.group(1))
        # Map to RAPTOR season (parquet 2024 -> 2024-25 season -> RAPTOR 2025)
        raptor_season = parquet_season + 1
        
        season_str = f"{parquet_season}-{str(parquet_season+1)[2:]}"
        
        # Get resolution map for this season
        if raptor_season not in resolution_cache:
            resolution_cache[raptor_season] = get_season_player_map(season_str)
            time.sleep(1.0) # Rate limit
            
        res_map = resolution_cache[raptor_season]
        
        print(f"\n🚀 Backfilling {os.path.basename(file_path)} (RAPTOR Season {raptor_season})...")
        df = pd.read_parquet(file_path)
        
        # We'll calculate the advantage for each game
        game_results = []
        
        for game_id, game_df in tqdm(df.groupby('GAME_ID')):
            game_df = game_df.sort_values('actionNumber')
            
            # 1. Identify Home/Away Teams
            h_score_diff = game_df['home_score'].diff().fillna(0)
            mask_h_scoring = (h_score_diff > 0) & (game_df['PLAYER1_TEAM_ID'].notna())
            home_team_id = game_df.loc[mask_h_scoring, 'PLAYER1_TEAM_ID'].iloc[0] if mask_h_scoring.any() else None
            
            team_ids = [tid for tid in game_df['PLAYER1_TEAM_ID'].unique() if pd.notna(tid) and tid != 0]
            if home_team_id is None and len(team_ids) >= 1: home_team_id = team_ids[0]
            
            away_team_id = None
            for tid in team_ids:
                if tid != home_team_id:
                    away_team_id = tid
                    break
            
            if not home_team_id or not away_team_id:
                # Fallback: fill with 0
                game_df['live_raptor_advantage'] = 0.0
                game_results.append(game_df)
                continue

            # 2. Offline Starter Inference
            # Starters are players who appear in an action before they are subbed IN.
            starters = {home_team_id: set(), away_team_id: set()}
            entered_names = {home_team_id: set(), away_team_id: set()}
            
            for _, row in game_df.iterrows():
                tid = int(row['PLAYER1_TEAM_ID']) if pd.notna(row['PLAYER1_TEAM_ID']) else 0
                if tid not in starters: continue
                
                desc = row['description'] if pd.notna(row['description']) else ""
                
                # Robust substitution detection
                is_sub = (
                    (row.get('EVENTMSGTYPE') == 8) or 
                    (normalize_name(row.get('actionType', '')) == 'substitution') or 
                    (normalize_name(row.get('EVENTMSGTYPE_STR', '')) == 'substitution')
                )
                
                if is_sub:
                    # Player going OUT (playerName)
                    out_name = normalize_name(row['playerName'])
                    if out_name not in entered_names[tid] and len(starters[tid]) < 5:
                        starters[tid].add(out_name)
                    
                    # Track player coming IN
                    match = sub_pattern.search(desc)
                    if match:
                        in_name = normalize_name(match.group(1))
                        entered_names[tid].add(in_name)
                else:
                    # Normal action
                    p_name = normalize_name(row['playerName'])
                    if p_name and p_name not in entered_names[tid] and len(starters[tid]) < 5:
                        starters[tid].add(p_name)
                        
            # Resolve starter names to full names for RAPTOR lookup
            def resolve_names(name_set, tid):
                full_names = set()
                for n in name_set:
                    full = res_map.get((n, tid), n) # Fallback to name if not found
                    full_names.add(normalize_name(full))
                return full_names

            home_on_floor = resolve_names(starters[home_team_id], home_team_id)
            away_on_floor = resolve_names(starters[away_team_id], away_team_id)
            
            # 3. Track Substitutions and Advantage
            advantages = []
            
            for _, row in game_df.iterrows():
                tid = int(row['PLAYER1_TEAM_ID']) if pd.notna(row['PLAYER1_TEAM_ID']) else 0
                
                # Robust substitution detection
                is_sub = (
                    (row.get('EVENTMSGTYPE') == 8) or 
                    (normalize_name(row.get('actionType', '')) == 'substitution') or 
                    (normalize_name(row.get('EVENTMSGTYPE_STR', '')) == 'substitution')
                )
                
                if is_sub and tid in [home_team_id, away_team_id]:
                    desc = row['description'] if pd.notna(row['description']) else ""
                    out_name = normalize_name(row['playerName'])
                    
                    match = sub_pattern.search(desc)
                    if match:
                        in_name_raw = match.group(1)
                        in_name_norm = normalize_name(in_name_raw)
                        in_full_name = normalize_name(res_map.get((in_name_norm, tid), in_name_raw))
                        
                        target_set = home_on_floor if tid == home_team_id else away_on_floor
                        # Try to remove out_name or its resolution
                        out_full_name = normalize_name(res_map.get((out_name, tid), out_name))
                        
                        if out_full_name in target_set:
                            target_set.remove(out_full_name)
                        elif out_name in target_set:
                            target_set.remove(out_name)
                        
                        target_set.add(in_full_name)
                
                # Calculate Advantage
                def get_total_raptor(player_set):
                    total = 0.0
                    for p in player_set:
                        # Use -1.0 as replacement level default
                        total += master_raptor.get((p, raptor_season), -1.0)
                    return total
                
                adv = get_total_raptor(home_on_floor) - get_total_raptor(away_on_floor)
                advantages.append(adv)
            
            game_df['live_raptor_advantage'] = advantages
            game_results.append(game_df)
            
        # Combine and Save
        df_updated = pd.concat(game_results)
        df_updated.to_parquet(file_path, index=False)
        print(f"  - Successfully updated {file_path}")

    print("\n✅ Master Backfill Complete!")

if __name__ == "__main__":
    backfill_raptor_advantage()
