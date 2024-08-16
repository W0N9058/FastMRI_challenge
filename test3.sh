#!/bin/bash

LOG_FILE="bebygan.log"

# 로그 파일 초기화
echo "" > $LOG_FILE

# 테스트할 조합
combinations=(
    "4 11 4"
)

# 작업 경로 설정
FASTMRI_PATH="$HOME/FastMRI_challenge"
BEBYGAN_PATH="$HOME/FastMRI_challenge/temp/Simple-SR-master/exps/BebyGAN"

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
        --sens_chans $sens_chans

    # train에 대한 reconstruct 코드 실행
    python reconstruct_for_bebygan.py \
        --cascade $cascade \
        --chans $chans \
        --sens_chans $sens_chans

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

    echo "Completed for Varnet cascade: $cascade, chans: $chans, sens_chans: $sens_chans" >> $LOG_FILE
                
    # Beby-Gan reconstruct 실행
    cd "$BEBYGAN_PATH"
    python reconstruct.py

    # 평가
    cd "$FASTMRI_PATH"
    python leaderboard_eval_modified.py \
        -lp "/home/Data/leaderboard" \
        -yp "../result/test_Varnet/reconstructions_leaderboard_processed/" \
        --log_file $LOG_FILE

    echo "Completed for Beby-GAN cascade: $cascade, chans: $chans, sens_chans: $sens_chans" >> $LOG_FILE
    echo "=====================================" >> $LOG_FILE
done
