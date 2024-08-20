FASTMRI_PATH="$HOME/FastMRI_challenge"
BEBYGAN_PATH="$HOME/FastMRI_challenge/temp/Simple-SR-master/exps/BebyGAN"

# Varnet train 코드 실행
python train.py \
    -b 1 \
    -e 3 \
    -l 0.0001 \
    -r 10 \
    -n "test_Varnet" \
    -t "/home/Data/train/" \
    -v "/home/Data/val/" \
    --cascade 4 \
    --chans 11 \
    --sens_chans 4 \
    --aug_schedule "ramp" \
    --aug_strength 1.0
                
# train에 대한 reconstruct 코드 실행
python reconstruct_for_bebygan.py
    --cascade 4 \
    --chans 11 \
    --sens_chans 4 

# Beby-Gan train 코드 실행
cd "$BEBYGAN_PATH"
python train.py