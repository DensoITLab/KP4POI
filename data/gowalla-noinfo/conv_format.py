import pandas as pd
import numpy as np
import pickle
import json
import os
import re
from tqdm import tqdm

def data_inter():
    thres = 3
    min_seq_len = 10
    user_id2name = {}
    user_id = []

    pois = []
    with open('sequential_data_all.txt','rt') as f:
        for line in f:
            a = line.rstrip().split(' ')
            uid = 'u'+a[0]
            user_id.append(uid)
            seq = a[1:]
            user_id2name[uid] = uid
            pois.extend(seq)
    
    pois = np.array(pois,dtype=int)

    with open('user_id2name.pkl','wb') as f:
        pickle.dump(user_id2name,f)

    u_poi, counts = np.unique(pois, return_counts=True)

    orig_count = len(u_poi)
    is_in = set(u_poi[counts>=thres])

    # count of items reduced from 2930451 to 699927
    print('count of {0} reduced from {2} to {1}'.format('items',len(is_in),orig_count))

    with open('items_in.pkl','wb') as f:
        pickle.dump(is_in,f)

    lines = []
    with open('sequential_data_all.txt','rt') as fseq:
        bar = tqdm(total = len(user_id))
        for seq in fseq:
            s = seq.rstrip().split(' ')
            uid = 'u'+s[0]
            s = [i for i in s[1:] if int(i) in is_in]
            bar.update(1)
            if len(s) < min_seq_len:
                print('# items {1} too small; skip {0}'.format(uid,len(s)))
                continue
            lines.append(uid+' '+' '.join(s))
    
    with open('sequential_data.txt','wt') as fseq2:
        fseq2.write('\n'.join(lines))


def data_item():
    datamaps = {}
    with open('user_id2name.pkl','rb') as f:
        user_id2name = pickle.load(f)

    datamaps['user2id'] = {str(v): str(k) for k, v in user_id2name.items()}
    datamaps['id2user'] = user_id2name

    with open('items_in.pkl','rb') as f:
        is_in = pickle.load(f)

    entity2text = {}

    item2entity = {}
    id2item = {}

    ## pandas skips some rows, so read them row by row
    with open('idx2poi.json', encoding='utf-8') as f:
        poi_info = json.load(f)
        for k in poi_info.keys():
            if int(k) in is_in:    
                eid = 'p'+k
                item2entity[eid] = eid
                id2item[k] = eid
 

    with open('item2entity.json','w') as f:
        json.dump(item2entity, f)

    with open('entity2text.json','w') as f:
        json.dump(entity2text, f, ensure_ascii=False)

    with open('triple2dict.json','w') as f:
        json.dump({}, f)

    datamaps['item2id'] = {str(v): str(k) for k, v in id2item.items()}
    datamaps['id2item'] = id2item

    with open('datamaps.json','w') as f:
        json.dump(datamaps, f)

if __name__ == '__main__':
    data_inter()
    data_item()

