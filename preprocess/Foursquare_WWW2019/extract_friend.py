import json
import pandas as pd
import networkx as nx
from utils import get_emb, kmeans, plot_pca, plot_cont_ratio
from sklearn.decomposition import PCA

def make_graph():
    user = {}
    friend = {}
    e_list = []
    with open('dataset_WWW_friendship_new.txt') as f:
        for line in f:
            u1, u2 = line.rstrip().split('\t')
            u1, u2 = int(u1), int(u2)
            e_list.append((u1,u2))
            if u1 not in friend:
                friend[u1] = [u2]
            else:
                friend[u1].append(u2)
            if u2 not in friend:
                friend[u2] = [u1]
            else:
                friend[u2].append(u1)
            
            user[u1] = 1
            user[u2] = 1
                
    with open('friend.json','wt') as fout:
        json.dump(friend,fout,indent=2)

    g = nx.Graph()
    g.add_nodes_from(user.keys())
    g.add_edges_from(e_list)
    print(f'# nodes = {g.number_of_nodes()}')
    print(f'# edges = {g.number_of_edges()}')

    return g


#g = make_graph()
#vec_df = get_emb(g)
#vec_df.to_csv('node_vec.csv')

vec_df = pd.read_csv('node_vec.csv',index_col=0).sort_index()
#labels = kmeans(vec_df)
#with open('community_kmeans.json','wt') as fout:
#    json.dump(labels,fout,indent=2)

dfs = vec_df.apply(lambda x: (x-x.mean())/x.std(), axis=0)
#print(dfs.head)

pca = PCA()
pca.fit(dfs)
feature = pca.transform(dfs)

pd.DataFrame(feature, columns=[f"PC{x}" for x in range(len(dfs.columns))], index=vec_df.index).to_csv('node_vec_pca.csv')


plot_pca(feature)
plot_cont_ratio(pca)


'''
c_part(g)
subgraphs = (g.subgraph(c) for c in nx.connected_components(g))
i=0
for sg in subgraphs:
    i+=1
    recursive_clustering(sg)
    assert 0
print(i)
'''

