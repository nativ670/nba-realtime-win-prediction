import time
from prefect import task, flow
from src.data_ingestion.fetch_box import run_box_ingestion
from src.data_ingestion.fetch_pbp import run_ingestion as run_pbp_ingestion
from src.features.build_dataset import build_training_dataset
from src.config import PROCESSED_DIR

@task(name="Fetch Box Scores", retries=2, retry_delay_seconds=60)
def task_fetch_box():
    print("\n[Phase 1/3] Fetching 10 years of box scores...")
    run_box_ingestion()

@task(name="Fetch Play-by-Play Data", retries=3, retry_delay_seconds=300)
def task_fetch_pbp():
    print("\n[Phase 2/3] Fetching 10 years of play-by-play data...")
    print("NOTE: This will take several hours due to nba_api rate limits.")
    run_pbp_ingestion()

@task(name="Build Training Dataset")
def task_build_dataset():
    print("\n[Phase 3/3] Engineering features and building final dataset...")
    build_training_dataset()

@flow(name="NBA Data Pipeline")
def main_flow():
    print("🏀 NBA Real-Time WP - Full Data Pipeline 🏀")
    print("===========================================")
    
    start_total = time.time()
    
    task_fetch_box()
    task_fetch_pbp()
    task_build_dataset()
    
    end_total = time.time()
    duration = (end_total - start_total) / 3600
    print(f"\n✅ Pipeline complete! Total time: {duration:.2f} hours.")
    print(f"Processed file: {PROCESSED_DIR / 'training_data.parquet'}")

if __name__ == "__main__":
    main_flow()
