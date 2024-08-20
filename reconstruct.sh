FASTMRI_PATH="$HOME/FastMRI_challenge"
BEBYGAN_PATH="$HOME/FastMRI_challenge/temp/Simple-SR-master/exps/BebyGAN"\

# Varnet reconstruct 실행
python reconstruct.py \
    -b 2 \
    -n "test_Varnet" \
    -p "/home/Data/leaderboard" \
    --cascade 4 \
    --chans 11 \
    --sens_chans 4
    
# Beby-Gan reconstruct 실행
cd "$BEBYGAN_PATH"
python reconstruct.py