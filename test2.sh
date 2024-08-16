#!/bin/bash

LOG_FILE="augmentation.log"

# 로그 파일 초기화
echo "" > $LOG_FILE

# 테스트할 조합
combinations=("2 9 5") # test1에서 우세한 파라미터

# aug_schedule 옵션
aug_schedules=("constant" "ramp" "exp")

# aug_strength 옵션
aug_strengths=("0.01" "0.1")

for combo in "${combinations[@]}"
do
    IFS=' ' read -r -a params <<< "$combo"
    cascade=${params[0]}
    chans=${params[1]}
    sens_chans=${params[2]}

    for aug_schedule in "${aug_schedules[@]}"
    do
        for aug_strength in "${aug_strengths[@]}"
        do
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
                --aug_schedule $aug_schedule \
                --aug_strength $aug_strength

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
    done
done