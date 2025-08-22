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
                    #entity2text[aid] = add
                    address[add] = aid
                    
                #triple[eid] = {'poi.poi.address':[address[add]]}

    feat_type = {}

    pca_dim = 5
    df = pd.read_csv('node_vec_pca.csv',index_col=0,header=0)
    d = df.to_numpy().astype(int)
    pca_feat = {}
    for i,j in enumerate(df.index):
        pca_feat[str(j)] = '_'.join([str(x) for x in d[i,0:pca_dim]])
    
    for uname in user_id2name.keys():
        uid = uname[1:]

        entity2text[uname] = uname
        item2entity[uname] = uname
        id2item[uname] = uname

        feat = str(pca_feat[uid])
        if feat not in feat_type:
            fid = 'feat='+feat
            entity2text[fid] = feat
            feat_type[feat] = fid
        
        triple[uname] = {'poi.user.nodevec':[feat_type[feat]]}

    print(f'address {len(address)} and feat_type {len(feat_type)}')
    
    # # poi 536810
    # # of address 500 and community 635

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

