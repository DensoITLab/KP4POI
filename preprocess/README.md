# download dataset

1. Five Foursquare dataset from https://sites.google.com/site/yangdingqi/home/foursquare-dataset
2. One gowalla dataset from https://snap.stanford.edu/data/loc-gowalla.html

# preprocessing raw data for sequential data with address
1. unzip downloaded zipfiles
2. Copy all files in the unzipped directories into the prepared corresponding Foursquare and gowalla directories

For each directory
1. run `extract_POI.py`
2. run `extract_address.py`
3. run `make_seq_data.py`, which will generate `pid2idx.json`, `id2poi.json`, and `user_history.txt` (these are required for KP4POI, so copy or symbolic link them to the data directory in the top directory)

# preprocessing social graph
For `Foursquare_WWW2019` and `gowalla`, friendship graphs are provided.
`extract_friend.py` will convert them into community ID `community.json`
(this is required for KP4POI with user knowledge prompting such as `Foursquare_WWW2019-w_add_u`)
