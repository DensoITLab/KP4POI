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

    category = {}
    address = {}

    pat = re.compile('(^[^\(]+) \((.+)\)$')

    ## pandas skips some rows, so read them row by row
    with open('idx2poi.json', encoding='utf-8') as f:
        poi_info = json.load(f)
        for k in poi_info.keys():
            if int(k) in is_in:    
                eid = 'p'+k
                m = pat.match(poi_info[k])
                if m:
                    a = m.groups()
                assert len(a) == 2, poi_info[k]
                add, cat = a[0], a[1]

                entity2text[eid] = add
                item2entity[eid] = eid
                id2item[k] = eid

                if cat not in category:
                    cid = 'c'+cat
                    #entity2text[cid] = cat
                    category[cat] = cid

                add2 = add
                if 'No.' in add:
                    add2 = add.split(' No.')[0]
                elif '丁目' in add:
                    add2 = add.split('丁目')[0]

                if cat not in category:
                    cid = 'c'+cat
                    entity2text[cid] = cat
                    category[cat] = cid
                
                if add2 not in address:
                    aid = 'a'+add2
                    entity2text[aid] = add2
                    address[add2] = aid
                

                triple[eid] = {'poi.poi.address':[address[add2]]} #'poi.poi.category':[category[cat]]}
    
    print(f'# of category {len(category)} and address {len(address)}')

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

