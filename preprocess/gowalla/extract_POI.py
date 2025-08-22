import pandas as pd
from datetime import datetime as dt

import locale

locale.setlocale(locale.LC_TIME, 'en_US.UTF-8')

dt1 = dt(year=2009, month=2, day=1, hour=0)

df = pd.read_csv('loc-gowalla_totalCheckins.txt',delimiter='\t',encoding='latin-1',names=['u_id','time','lat','long','poi_id'],header=None)

#2010-10-19T23:55:27Z
df['time'] = df['time'].map(lambda x: dt.strptime(x,'%Y-%m-%dT%H:%M:%SZ'))
df['elapsed time'] = df['time'].map(lambda x: int((x-dt1).total_seconds()))

print(df.head())

df = df.sort_values('elapsed time')

df.to_csv('dataset_gowalla.tsv',sep='\t')

lat_long = {}
for lat, long in zip(df['lat'],df['long']):
    s = '%.3f=%.3f' % (lat, long)
    lat_long[s] = 1

with open('lat_long.txt','wt') as fout:
    for k in lat_long.keys():
        fout.write(k+'\n')

