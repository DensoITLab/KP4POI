# KP4POI
Knowledge prompting for point of interest (POI) recommendation.

![KP4POI](image/LLM_recom.png)

This is based on KP4SR.
1. Clone KP4SR, which is our baseline model `https://github.com/zhaijianyang/kp4sr`
2. Update files in `scripts` and `src` directories

# knowledge prompt for POI recommendation

## masked personalized prompt
![MPP](image/mpp.png)
## venue knowledge prompt
![kp_venue](image/kp_venue.png)
## user knowledge prompt
![kp_user](image/kp_user.png)


# data preprocess
1. convert and add information to the original data
    - See README.md in the `preprocess` directory
2. convert data required by KP4SR
    - See README.md in the `data` directory for experiment

# Model training and testing
After data are prepared, `bash run_poi.sh`

# citation
If you find this useful, please cite our paper.
```
@inproceedings{
author = {Yuuki Tachioka},
title = {{KP4POI}: Efficient POI recommendation on large-scale datasets via knowledge prompting of venues and users},
booktitle = {Proceedings of ACM RecSys Workshop on Recommenders in Tourism @ 19th ACM Conference on Recommender Systems (RecSys 2025)},
location = {Prague, Czech Republic},
month = {9},
year = {2025},
}
```
