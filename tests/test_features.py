import pytest
from src.features.pregame import haversine_vectorized
import numpy as np

def test_haversine_vectorized():
    # Boston (TD Garden) to LA (Crypto.com Arena)
    # BOS: 42.366, -71.062
    # LAL: 34.043, -118.267
    bos_lat = np.array([42.366])
    bos_lon = np.array([-71.062])
    lal_lat = np.array([34.043])
    lal_lon = np.array([-118.267])
    
    dist = haversine_vectorized(bos_lat, bos_lon, lal_lat, lal_lon)[0]
    
    # Distance is approx 2588 miles. Allow ~5% error for haversine radius assumptions.
    assert 2500 < dist < 2700

def test_same_location_distance():
    lat = np.array([42.366])
    lon = np.array([-71.062])
    
    dist = haversine_vectorized(lat, lon, lat, lon)[0]
    assert dist == 0.0
