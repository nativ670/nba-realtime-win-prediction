import pandas as pd
import numpy as np
import os
import sys
import time
from datetime import datetime, timedelta

# --- Path Injection ---
# Add the project root to sys.path so 'src' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.data_ingestion.update_elo import update_elo
from src.data_ingestion.update_pbp import update_pbp
from src.features.raptor_engine import run_raptor_pipeline

def run_daily_update():
    """
    Orchestrates the daily update of Elo, PBP, and RAPTOR data.
    Ensures that Elo is updated first so that PBP can use the latest ratings.
    """
    print("🚀 Starting Daily NBA Data Update...")
    start_time = time.time()

    # 1. Update Team Elo Ratings (Recursive, must be done sequentially by date)
    print("\n[Step 1/3] Updating Team Elo Ratings...")
    try:
        update_elo()
    except Exception as e:
        print(f"❌ Error updating Elo: {e}")
        # We might want to continue or stop depending on how critical Elo is for PBP
        # In this project, PBP uses Elo for feature engineering, so we should be careful.

    # 2. Update Play-by-Play Data (Uses the updated Elo file)
    print("\n[Step 2/3] Updating Play-by-Play Data...")
    try:
        update_pbp()
    except Exception as e:
        print(f"❌ Error updating PBP: {e}")

    # 3. Update Player RAPTOR Ratings (Cumulative for current season)
    print("\n[Step 3/3] Updating Player RAPTOR Ratings...")
    try:
        run_raptor_pipeline()
    except Exception as e:
        print(f"❌ Error updating RAPTOR: {e}")

    end_time = time.time()
    duration = (end_time - start_time) / 60
    print(f"\n✅ Daily update complete! Total time: {duration:.2f} minutes.")

if __name__ == "__main__":
    run_daily_update()
