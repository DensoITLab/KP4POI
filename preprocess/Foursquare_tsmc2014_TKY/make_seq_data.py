import json

address = {}
with open('lat_long_TKY_add.jsonl') as f:
    for line in f:
        d = json.loads(line.rstrip())
        address[d['Feature'][0]['Geometry']['Coordinates'].replace(',','=')] = d['Feature'][0]['Property']['Address']

user_history = {}
pid2idx = {}
idx2poi = {}

with open('dataset_TSMC2014_TKY.tsv') as f:
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
            lon_lat = '%.3f=%.3f' % (float(d[6]), float(d[5]))
            lat_lon = '%.3f=%.3f' % (float(d[5]), float(d[6]))

            poi = '%s (%s)' % (address[lon_lat],cat)
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

