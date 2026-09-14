from fastapi.testclient import TestClient
from src.models.api import app, GameFeatureList, EnsemblePayload

client = TestClient(app)

def test_predict_win_prob():
    # Use Pydantic models to construct payload
    seq = EnsemblePayload(
        sequence=[
            GameFeatureList(
                score_differential=2,
                seconds_remaining_in_game=2880.0,
                is_home_possession=1.0,
                elo_advantage=15.0,
                rest_advantage=0.0,
                distance_traveled=0.0,
                momentum_differential=0.0,
                home_timeouts_remaining=7,
                away_timeouts_remaining=7,
                home_in_bonus=0,
                away_in_bonus=0,
                live_raptor_advantage=1.5
            ),
            GameFeatureList(
                score_differential=4,
                seconds_remaining_in_game=2850.0,
                is_home_possession=1.0,
                elo_advantage=15.0,
                rest_advantage=0.0,
                distance_traveled=0.0,
                momentum_differential=1.2,
                home_timeouts_remaining=7,
                away_timeouts_remaining=7,
                home_in_bonus=0,
                away_in_bonus=0,
                live_raptor_advantage=1.8
            )
        ]
    )
    payload = seq.model_dump()

    # The API loads models on startup via lifespan context manager.
    # FastAPI's TestClient triggers lifespan automatically when used in a with block.
    with TestClient(app) as client:
        response = client.post("/predict_win_prob", json=payload)
        
        assert response.status_code == 200, f"Error: {response.text}"
        result = response.json()
        
        expected_keys = ["xgb_prob", "lstm_prob", "final_win_probability"]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        
        assert 0.0 <= result["final_win_probability"] <= 1.0
