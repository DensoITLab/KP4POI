import pandas as pd

u_id_train = []
u_id_test = []

pos_train = []
pos_test = []

n_users = 0
n_items = 0

with open('user_history.txt') as f:
    for line in f:
        d = line.rstrip().split(' ')
        u_id = int(d[0])

        if n_users < u_id:
            n_users = u_id

        seq_len = len(d)-1

        for i in range(1,len(d)):
            if n_items < int(d[i]):
                n_items = int(d[i])

        n_test = max(seq_len//(6*10),1)

        d_train = [d[i] for i in range(1,len(d)-(n_test*6))]
        d_test = [d[i] for i in range(len(d)-(n_test*6)+1,len(d))]

        u_id_train += [u_id]*len(d_train)
        pos_train += d_train

        u_id_test += [u_id]*len(d_test)
        pos_test += d_test

n_users += 1
n_items += 1
with open('info.yaml','wt') as fout:
    fout.write(f'n_items: {n_items}\n')
    fout.write(f'n_users: {n_users}\n')

df_train = pd.DataFrame([u_id_train,pos_train]).T
df_train.columns=['uid','loc']

df_test = pd.DataFrame([u_id_test,pos_test]).T
df_test.columns=['uid','loc']

df_train.to_csv('train.csv')
df_test.to_csv('test.csv')
