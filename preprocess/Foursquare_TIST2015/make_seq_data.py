import numpy as np
import json
from city import city


# https://amandinancy16.medium.com/reverse-geocoding-with-geopy-c26cfb63f74c

address = np.load('lat_long_add.npy', allow_pickle=True).item()


user_history = {}
pid2idx = {}
idx2poi = {}

with open('dataset_TIST2015.tsv') as f:
    i = 0
    prev_etime = 0
    for line in f:
        i += 1
        if i > 1:
            d = line.rstrip().split('\t')
            u_id = int(d[1])-1 # 0 start
            assert u_id >= 0
            poi_id = d[2]
            cat = d[8]
            lat_lon = '%.3f=%.3f' % (float(d[6]), float(d[7]))

            poi = '%s (%s)' % (address[lat_lon].name,cat)
            etime = int(d[5])

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

