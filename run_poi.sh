#!/bin/bash

declare -A info_types=(
    ["Foursquare_tsmc2024_NYC"]="noinfo w_add w_latlon w_add_latlon w_cat w_cat_add w_cat_latlon w_cat_add_latlon"
    ["Foursquare_tsmc2024_TKY"]="noinfo w_add w_latlon w_add_latlon w_cat w_cat_add w_cat_latlon w_cat_add_latlon"
    ["Foursquare_tsmc2024_NYC-TKY"]="noinfo w_add w_latlon w_add_latlon w_cat w_cat_add w_cat_latlon w_cat_add_latlon"
    ["Foursquare_TIST2015"]="noinfo w_add w_cat w_cat_add w_cat_latlon w_cat_add_latlon"
    ["Foursquare_WWW2019"]="noinfo w_add w_cat w_u2 w_u3 w_cat_add w_cat_latlon w_cat_add_latlon w_cat_add_u w_cat_add_u2"
    ["gowalla"]="noinfo w_add w_latlon w_add_latlon w_u2 w_u3 w_add_u w_add_u2"
)


test_prompt=2-11
#test_prompt=2-6

prep() {
    cd data/${task}
    python conv_format.py
    cd -
}

train_conditioned() {
    outd=out/${task}_t5-small_2hop_d8_L5_1e-3_mask
    for i in ${info_types[$task]}; do
        pushd data
        if [ -e ${task} ]; then
            rm ${task}
        fi
        ln -s ${task}-${i} ${task}
        pushd ${task}
        python conv_format.py
        popd
        popd
        bash scripts/modelarts_train.sh ${task}
        mkdir ${outd}/${i}
        mv ${outd}/BEST_EVAL_LOSS{_opt.pth,.pth} ${outd}/train.log ${outd}/${i}/ || exit 1
        bash scripts/modelarts_test.sh ${task} ${test_prompt} ${outd}/${i}
    done
}

eval_res() {
    outd=out/${task}_t5-small_2hop_d8_L5_1e-3_mask
    python scripts/evaluate.py \
        ${outd}/noinfo/result_${test_prompt}.json
    for i in ${info_types_all[$task]}; do
        if [ ${i} != 'noinfo' ]; then
            python scripts/evaluate.py \
                ${outd}/${i}/result_${test_prompt}.json --t-test ${outd}/noinfo/result_${test_prompt}.json
        fi
    done
}



for task in Foursquare_tsmc2024_{TKY,NYC,NYC-TKY} Foursquare_TIST2015 Foursquare_WWW2019 gowalla; do
    prep
    train_conditioned
    eval_res > out/result_${task}.tsv
done

