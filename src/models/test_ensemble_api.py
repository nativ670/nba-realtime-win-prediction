import requests
import json

def test_ensemble_api():
    url = "http://127.0.0.1:8000/predict_win_prob"
    
    # Sample payload with a sequence of length 2
    # The API will pad this to length 15 automatically
    payload = {
        "sequence": [
            {
                "score_differential": 2,
                "seconds_remaining_in_game": 2880.0,
                "possession_team_id": 1610612737,
                "elo_advantage": 15.0,
                "rest_advantage": 0,
                "distance_traveled": 0.0,
                "momentum_differential": 0.0,
                "home_timeouts_remaining": 7,
                "away_timeouts_remaining": 7,
                "home_in_bonus": 0,
                "away_in_bonus": 0
            },
            {
                "score_differential": 4,
                "seconds_remaining_in_game": 2850.0,
                "possession_team_id": 1610612737,
                "elo_advantage": 15.0,
                "rest_advantage": 0,
                "distance_traveled": 0.0,
                "momentum_differential": 1.2,
                "home_timeouts_remaining": 7,
                "away_timeouts_remaining": 7,
                "home_in_bonus": 0,
                "away_in_bonus": 0
            }
        ]
    }

    print(f"🚀 Sending request to {url}...")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        result = response.json()
        print("\n✅ API Response Received:")
        print(json.dumps(result, indent=4))
        
        # Verify keys
        expected_keys = ["xgb_prob", "lstm_prob", "final_win_probability"]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        
        print("\n✨ All checks passed!")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to the API. Is it running?")
        print("Run 'python src/models/api.py' in another terminal.")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")

if __name__ == "__main__":
    test_ensemble_api()
