#!/bin/bash

LOG_FILE="result.log"

# 로그 파일 초기화
echo "" > $LOG_FILE

# 테스트할 조합
combinations=(
    "2 9 5"
    "4 11 4"
)

# aug_schedule 옵션
aug_schedules=("constant" "ramp" "exp")

# aug_strength 옵션
aug_strengths=("10.0" "5.0" "1.0" "0.5" "0.1" "0.05" "0.01")

# 작업 경로 설정
FASTMRI_PATH="$HOME/root/FastMRI_challenge"
BEBYGAN_PATH="$HOME/temp/Simple-SR-master/exps/BebyGAN"

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
                
            # train에 대한 reconstruct 코드 실행
            python reconstruct_for_bebygan.py

            # Beby-Gan train 코드 실행
            cd "$BEBYGAN_PATH"
            python train.py

            # 재구성
            cd "$FASTMRI_PATH"
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
                
            # Beby-Gan reconstruct 실행
            cd "$BEBYGAN_PATH"
            python reconstruct.py

            # 평가
            cd "$FASTMRI_PATH"
            python leaderboard_eval_modified.py \
                -lp "/home/Data/leaderboard" \
                -yp "../result/test_Varnet_cascade/reconstructions_leaderboard_processed/" \
                --log_file $LOG_FILE

            echo "Completed for Beby-GAN cascade: $cascade, chans: $chans, sens_chans: $sens_chans, aug_schedule: $aug_schedule, aug_strength: $aug_strength" >> $LOG_FILE
            echo "=====================================" >> $LOG_FILE
        done
    done
done