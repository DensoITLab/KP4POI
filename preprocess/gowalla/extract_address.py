import json
import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.neighbors import NearestNeighbors
from city import city

# 最近傍探索でcity nameを取り出す
# https://qiita.com/wasnot/items/20c4f30a529ae3ed5f52

df_city = pd.read_csv('dataset_WWW2019_Cities.txt',delimiter='\t',encoding='latin-1',names=['city','lat','long','country code','country','city type'],header=None,index_col=None)
d = df_city.to_dict()

print(f"number of cities {len(d['city'])}")

xb = np.zeros((len(d['city']),2))

cities = []
for i in range(len(d['city'])):
    c = city()
    c.name = d['city'][i]
    c.country = d['country'][i]
    c.country_code = d['country code'][i]
    c.city_type = d['city type'][i]
    cities.append(c)
    xb[i,0] = float(d['lat'][i])
    xb[i,1] = float(d['long'][i])

nn = NearestNeighbors(metric='minkowski',p=2) # euclidian distance
nn.fit(xb)

result = {}

coo = []

with open('lat_long.txt') as f:
    for line in f:
        (lat,lon) = line.rstrip().split('=')
        coo.append([float(lat),float(lon)])

coo = np.array(coo)
dists, neigh = nn.kneighbors(coo, n_neighbors=1)

with open('lat_long.txt') as f:
    i = 0
    for line in tqdm(f,total=coo.shape[0]):
        line = line.rstrip()
        (lat,lon) = line.split('=')
        found = False
        for j in range(neigh.shape[1]):
            c = cities[neigh[i,j]]
            result[lat+'='+lon] = c

        i += 1

print(f'number of lines {i}')
print(f'number of results {len(result)}')

np.save('lat_long_add.npy',result, allow_pickle=True)

