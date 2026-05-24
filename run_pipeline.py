import os
import sys
import time


from src.data_ingestion.fetch_box import run_box_ingestion
from src.data_ingestion.fetch_pbp import run_ingestion as run_pbp_ingestion
from src.features.build_dataset import build_training_dataset

def main():
    print("🏀 NBA Real-Time WP - Full Data Pipeline 🏀")
    print("===========================================")
    try:
        start_total = time.time()
    
        # 1. Fetch Box Scores
        print("\n[Phase 1/3] Fetching 10 years of box scores...")
        run_box_ingestion()
        
        # 2. Fetch Play-by-Play
        print("\n[Phase 2/3] Fetching 10 years of play-by-play data...")
        print("NOTE: This will take several hours due to nba_api rate limits.")
        run_pbp_ingestion()
        
        # 3. Build Dataset
        print("\n[Phase 3/3] Engineering features and building final dataset...")
        build_training_dataset()
        
        end_total = time.time()
        duration = (end_total - start_total) / 3600
        from src.config import PROCESSED_DIR
        print(f"\n✅ Pipeline complete! Total time: {duration:.2f} hours.")
        print(f"Processed file: {PROCESSED_DIR / 'training_data.parquet'}")
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main()
