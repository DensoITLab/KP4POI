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

    with open('idx2poi.json', encoding='utf-8') as f:
        poi_info = json.load(f)
        for k in poi_info.keys():
            if int(k) in is_in:    
                eid = 'p'+k
                add = poi_info[k]

                entity2text[eid] = add
                item2entity[eid] = eid
                id2item[k] = eid
              
                if add not in address:
                    aid = 'a'+add
                    entity2text[aid] = add
                    address[add] = aid
                

                triple[eid] = {'poi.poi.address':[address[add]]}
    
    print(f'# of address {len(address)}')

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

