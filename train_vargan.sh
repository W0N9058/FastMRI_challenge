# varnet train 코드 실행
python train.py \
  -b 1 \
  -e 5 \
  -l 0.0001 \
  -r 10 \
  -n 'test_Varnet' \
  -t '/home/Data/train/' \
  -v '/home/Data/val/' \
  --cascade 1 \
  --chans 9 \
  --sens_chans 4

# train에 대한 reconstruct 코드 실행
python reconstruct_for_bebygan.py

# Beby-Gan train 코드 실행
cd temp/Simple-SR-master/exps/BebyGAN

python train.py

