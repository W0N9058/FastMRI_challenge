#!/bin/bash

LOG_FILE="parameter.log"

# 로그 파일 초기화
echo "" > $LOG_FILE

# 테스트할 조합
combinations=("2 9 5" "4 11 4")

## aug_strength 0.0으로 설정해서 실행

for combo in "${combinations[@]}"
do
    IFS=' ' read -r -a params <<< "$combo"
    cascade=${params[0]}
    chans=${params[1]}
    sens_chans=${params[2]}

    # 모델 학습
    python train.py \
        -b 1 \
        -e 3 \
        -l 0.0001 \
        -r 10 \
        -n "test_Varnet" \
        -t "/home/Data/train/" \
        -v "/home/Data/val/" \
        --cascade $cascade \
        --chans $chans \
        --sens_chans $sens_chans \
                
    # 재구성
    python reconstruct_modified.py \
        -b 2 \
        -n "test_Varnet" \
        -p "/home/Data/leaderboard" \
        --cascade $cascade \
        --chans $chans \
        --sens_chans $sens_chans \
        --log_file $LOG_FILE
                
    # 평가
    python leaderboard_eval_modified.py \
        -lp "/home/Data/leaderboard" \
        -yp "../result/test_Varnet/reconstructions_leaderboard/" \
        --log_file $LOG_FILE

    echo "Completed for Varnet cascade: $cascade, chans: $chans, sens_chans: $sens_chans, aug_schedule: $aug_schedule, aug_strength: $aug_strength" >> $LOG_FILE
    echo "=====================================" >> $LOG_FILE
done
