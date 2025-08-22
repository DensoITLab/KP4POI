# interaction data preparation

1. Copy or symbolic link `user_history.txt`, `id2poi.json`, and `pid2idx.json` in the preprocess directory to the 
```
Foursquare_TIST2015-w_cat
Foursquare_tsmc2024_NYC-TKY-w_cat
Foursquare_tsmc2024_NYC-w_cat
Foursquare_tsmc2024_TKY-w_cat
Foursquare_WWW2019-w_cat
gowalla-noinfo
```
Note that just **replace** the dummy `id2poi.json` and `pid2idx.json` with the correct files.

2. Rename `user_history.txt` to `sequential_data_all.txt`
Dummy files are provided in the dataset and symbolic links of them are provided in other directories, since the size of interaction data is large but this is common for all experiments

3. Run `conv_format.py` in each directory

