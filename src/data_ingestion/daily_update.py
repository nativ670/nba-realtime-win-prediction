import os
import sys
import time
from pathlib import Path

# --- Path Injection ---
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
    start_time_total = time.time()

    # 1. Update Team Elo Ratings
    print("\n[Step 1/3] Updating Team Elo Ratings...")
    start_time_elo = time.time()
    try:
        update_elo()
        print(f"✅ Elo update finished in {(time.time() - start_time_elo):.2f}s")
    except Exception as e:
        print(f"❌ CRITICAL Error updating Elo: {e}")
        print("Elo update is required for PBP features. Exiting pipeline.")
        sys.exit(1)

    # 2. Update Play-by-Play Data
    print("\n[Step 2/3] Updating Play-by-Play Data...")
    start_time_pbp = time.time()
    try:
        update_pbp()
        print(f"✅ PBP update finished in {(time.time() - start_time_pbp):.2f}s")
    except Exception as e:
        print(f"❌ Error updating PBP: {e}")
        import traceback
        traceback.print_exc()

    # 3. Update Player RAPTOR Ratings
    print("\n[Step 3/3] Updating Player RAPTOR Ratings...")
    start_time_raptor = time.time()
    try:
        run_raptor_pipeline()
        print(f"✅ RAPTOR update finished in {(time.time() - start_time_raptor):.2f}s")
    except Exception as e:
        print(f"❌ Error updating RAPTOR: {e}")
        import traceback
        traceback.print_exc()

    duration = (time.time() - start_time_total) / 60
    print(f"\n✅ Daily update complete! Total time: {duration:.2f} minutes.")

if __name__ == "__main__":
    run_daily_update()
