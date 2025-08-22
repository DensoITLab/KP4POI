import requests
import json
import numpy as np

from tqdm import tqdm
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# https://amandinancy16.medium.com/reverse-geocoding-with-geopy-c26cfb63f74c

def NYC():

    result = {}

    with open('lat_long_NYC.txt') as f:
        for line in tqdm(f):
            (lat,lon) = line.rstrip().split('=')

            coordinates = lat + " , " + lon

            geolocator = Nominatim(user_agent="ytachioka", timeout= 10)
            rgeocode = RateLimiter(geolocator.reverse, min_delay_seconds=0.1)
            address = rgeocode(coordinates)

            result[lat+'='+lon] = address

    np.save('lat_long_NYC_add.npy',result, allow_pickle=True)



NYC()

