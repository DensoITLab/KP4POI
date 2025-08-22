import pandas as pd
import numpy as np
import pickle
import json
import os
import re
from tqdm import tqdm
  

def data_item():
    datamaps = {}
    with open('user_id2name.pkl','rb') as f:
        user_id2name = pickle.load(f)

    datamaps['user2id'] = {str(v): str(k) for k, v in user_id2name.items()}
    datamaps['id2user'] = user_id2name

    with open('items_in.pkl','rb') as f:
        is_in = pickle.load(f)
    
    print(f'# poi {len(is_in)}')

    entity2text = {}

    item2entity = {}
    id2item = {}
    triple = {}

    address = {}
    lat_lon = {}

    pat = re.compile('(^[^\(]+) \((.+)\)$')

    with open('pid2idx.json', encoding='utf-8') as f:
        poi_info = json.load(f)
        for pid in poi_info.keys():
            k = poi_info[pid]['idx']

            if int(k) in is_in:    
                eid = 'p'+k
                add = poi_info[pid]['poi']

                entity2text[eid] = add
                item2entity[eid] = eid
                id2item[k] = eid

                if add not in address:
                    aid = 'a'+add
                    #entity2text[aid] = add2
                    address[add] = aid
                
                lat_lon_ = poi_info[pid]['lat_lon'].replace('=',' and ')

                if lat_lon_ not in lat_lon:
                    lid = 'll'+poi_info[pid]['lat_lon']
                    entity2text[lid] = lat_lon_
                    lat_lon[lat_lon_] = lid

                triple[eid] = {'poi.poi.lat_lon':[lat_lon[lat_lon_]]} # 'poi.poi.address':[address[add2]]}
    
    print(f'# of address {len(address)}, latlon {len(lat_lon)}')

    with open('item2entity.json','w') as f:
        json.dump(item2entity, f)

    with open('entity2text.json','w') as f:
        json.dump(entity2text, f, ensure_ascii=False)
    
    with open('triple2dict.json','w') as f:
        json.dump(triple, f, indent=2, ensure_ascii=False)

    datamaps['item2id'] = {str(v): str(k) for k, v in id2item.items()}
    datamaps['id2item'] = id2item

    with open('datamaps.json','w') as f:
        json.dump(datamaps, f)


data_item()

