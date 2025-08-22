import numpy as np
import json

address = {}

# https://amandinancy16.medium.com/reverse-geocoding-with-geopy-c26cfb63f74c


result = np.load('lat_long_NYC_add.npy', allow_pickle=True).item()

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


user_history = {}
pid2idx = {}
idx2poi = {}

with open('dataset_TSMC2014_NYC.tsv') as f:
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

