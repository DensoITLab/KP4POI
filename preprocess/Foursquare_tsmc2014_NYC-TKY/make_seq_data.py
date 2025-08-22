import numpy as np
import json
import pandas as pd

address = {}

# https://amandinancy16.medium.com/reverse-geocoding-with-geopy-c26cfb63f74c


def read_NYC_add():
    result = np.load('../Foursquare_tsmc2014_NYC/lat_long_NYC_add.npy', allow_pickle=True).item()

    address = {}

    for k in result.keys():
        a = result[k].raw['address']
        if 'road' in a:
            address[k] = a['road']
        elif 'neighbourhood' in a:
            address[k] = a['neighbourhood']
        elif 'town' in a:
            address[k] = a['town']
        elif 'hamlet' in a:
            address[k] = a['hamlet']
        elif 'city' in a:
            address[k] = a['city']
        elif 'village' in a:
            address[k] = a['village']
        else:
            assert 0, a

        if 'house_number' in a:
            address[k] += ' No.'+a['house_number']
    return address

def read_TKY_add():
    address = {}
    with open('../Foursquare_tsmc2014_TKY/lat_long_TKY_add.jsonl') as f:
        for line in f:
            d = json.loads(line.rstrip())
            lon_lat = d['Feature'][0]['Geometry']['Coordinates'].split(',')
            lat_lon = lon_lat[1]+'='+lon_lat[0]

            address[lat_lon] = d['Feature'][0]['Property']['Address']
    
    return address

def concate_list():
    with open('dataset_TSMC2014_NYC-TKY.tsv','wt') as fout:
        #	u_id	poi_id	cat_id	cat	lat	long	time offset	time(UTC)	elapsed time
        max_id = 0
        with open('../Foursquare_tsmc2014_NYC/dataset_TSMC2014_NYC.tsv') as f:
            i = 0
            j = 0
            for line in f:
                if i == 0:
                    fout.write(line)
                else:
                    d = line.split('\t')
                    d[0] = str(j)
                    if max_id < int(d[1]):
                        max_id = int(d[1])
                    fout.write('\t'.join(d))
                    j += 1
                i+=1

        with open('../Foursquare_tsmc2014_TKY/dataset_TSMC2014_TKY.tsv') as f:
            i = 0
            for line in f:
                if i != 0:
                    d = line.split('\t')
                    d[0] = str(j)
                    d[1] = str(int(d[1])+max_id) # 1 start
                    fout.write('\t'.join(d))
                    j += 1
                i+=1

    df = pd.read_csv('dataset_TSMC2014_NYC-TKY.tsv',delimiter='\t',encoding='latin-1',header=0,index_col=0)
    df = df.sort_values('elapsed time')

    df.to_csv('dataset_TSMC2014_NYC-TKY.tsv',sep='\t')



concate_list()

address = read_NYC_add()
print(len(address))
address |= read_TKY_add()
print(len(address))


user_history = {}
pid2idx = {}
idx2poi = {}

with open('dataset_TSMC2014_NYC-TKY.tsv') as f:
    i = 0
    prev_etime = 0
    for line in f:
        i += 1
        if i > 1:
            d = line.rstrip().split('\t')
            u_id = int(d[1])-1 # 0 start
            assert u_id >= 0
            poi_id = d[2]
            cat = d[4]
            lat_lon = '%.3f=%.3f' % (float(d[5]), float(d[6]))

            poi = '%s (%s)' % (address[lat_lon],cat)
            etime = int(d[9])

            if poi_id not in pid2idx:
                idx = str(len(pid2idx)) # 9 start
                pid2idx[poi_id] = {'idx': idx,'poi': poi,'lat_lon':lat_lon}
                idx2poi[idx] = poi

            assert etime - prev_etime >= 0, f'time {prev_etime} > {etime} at {i}'
            prev_etime = etime

            #print(pid2idx)

            if u_id not in user_history:
                user_history[u_id] = [pid2idx[poi_id]['idx']]
            else:
                user_history[u_id].append(pid2idx[poi_id]['idx'])
            

with open('pid2idx.json','wt') as fout:
    json.dump(pid2idx,fout,ensure_ascii=False)

with open('idx2poi.json','wt') as fout:
    json.dump(idx2poi,fout,ensure_ascii=False)


with open('user_history.txt','wt') as fout:
    for u_id in sorted(user_history.keys()):
        fout.write(str(u_id)+' ')
        fout.write(' '.join(user_history[u_id]))
        fout.write('\n')

