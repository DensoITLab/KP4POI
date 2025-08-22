import requests
import json

def TKY():
    url = 'https://map.yahooapis.jp/geoapi/V1/reverseGeoCoder'
    client_id = 'dj00aiZpPXZYaXVjWTVIN3psNSZzPWNvbnN1bWVyc2VjcmV0Jng9Zjg-'

    result = []

    with open('lat_long_TKY.txt') as f:
        for line in f:
            (lat,lon) = line.rstrip().split('=')

            params = {'lat': lat, 'lon': lon, 'appid': client_id, 'output': 'json'}

            response = requests.get(url, params=params)

            if response.status_code == 200:
                result.append(json.loads(response.text))
                print(result[-1])
            else:
                print(f'error {response.status_code}')
                break

    with open('lat_long_TKY_add.jsonl','w') as fout:
        for r in result:
            fout.write(json.dumps(r,ensure_ascii=False)+'\n')

