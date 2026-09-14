import pytest
from src.features.pregame import haversine_vectorized

def test_haversine_vectorized():
    # Boston (TD Garden) coordinates
    bos_lat, bos_lon = 42.3663, -71.0622
    # New York (Madison Square Garden) coordinates
    nyk_lat, nyk_lon = 40.7505, -73.9934
    
    # Distance should be roughly 185-195 miles
    distance = haversine_vectorized(bos_lat, bos_lon, nyk_lat, nyk_lon)
    
    assert 180 <= distance <= 200, f"Distance {distance} out of expected bounds"
