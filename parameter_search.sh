#!/bin/bash

LOG_FILE="result.log"

# 로그 파일 초기화
echo "" > $LOG_FILE

for cascade in $(seq 1 1)
do
  for chans in $(seq 8 8)
  do
    for sens_chans in $(seq 3 3)
    do
      # 모델 학습
      python train.py \
        -b 1 \
        -e 1 \
        -l 0.0001 \
        -r 10 \
        -n "test_Varnet_cascade${cascade}_chans${chans}_senschans${sens_chans}" \
        -t "/home/Data/train/" \
        -v "/home/Data/val/" \
        --cascade $cascade \
        --chans $chans \
        --sens_chans $sens_chans

      # 재구성
      python reconstruct_modified.py \
        -b 2 \
        -n "test_Varnet_cascade${cascade}_chans${chans}_senschans${sens_chans}" \
        -p "/home/Data/leaderboard" \
        --cascade $cascade \
        --chans $chans \
        --sens_chans $sens_chans \
        --log_file $LOG_FILE

      # 평가
      python leaderboard_eval_modified.py \
        -lp "/home/Data/leaderboard" \
        -yp "../result/test_Varnet_cascade${cascade}_chans${chans}_senschans${sens_chans}/reconstructions_leaderboard/" \
        --log_file $LOG_FILE

      echo "Completed cascade: $cascade, chans: $chans, sens_chans: $sens_chans" >> $LOG_FILE
      echo "=====================================" >> $LOG_FILE
    done
  done
done