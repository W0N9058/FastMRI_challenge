import cv2
import os
import numpy as np
import sys
import h5py
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + '/../..')

import torch

from utils.common import tensor2img

def process_and_save_h5(model, input_h5_path, output_h5_path, device):
    with torch.no_grad():
        with h5py.File(input_h5_path, 'r') as f:
            recon_data = f['reconstruction'][:]  # (N, 384, 384)
            target_data = f['target'][:]  # Read target data but do not modify
            
            processed_data = []
            
            for i in range(recon_data.shape[0]):
                lr_img = torch.tensor(recon_data[i:i+1, :, :]).unsqueeze(0).to(device)  # Add batch and channel dims
                output = model.G(lr_img)
                output_img = tensor2img(output).astype(np.float32)
                
                processed_data.append(output_img)
            
            processed_data = np.stack(processed_data, axis=0)
        
        # Save the modified reconstruction data to a new h5 file
        with h5py.File(output_h5_path, 'w') as f:
            f.create_dataset('reconstruction', data=processed_data)
            f.create_dataset('target', data=target_data)  # Copy the target data unchanged
            
        print(f'Saved processed data to {output_h5_path}')

if __name__ == '__main__':
    from config import config
    from network import Network
    from utils.model_opr import load_model

    # Model and configuration setup
    config.VAL.DATASETS = ['FASTMRI']
    config.VAL.SAVE_IMG = True

    model = Network(config)
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model.G = model.G.to(device)

    model_path = '/root/FastMRI_challenge/temp/Simple-SR-master/exps/BebyGAN/log/models/6000_G.pth'
    load_model(model.G, model_path, cpu=True)

    # Input and output directories
    input_dir = '/root/result/test_Varnet/reconstructions_leaderboard/'
    output_dir = '/root/result/test_Varnet/reconstructions_leaderboard_processed/'

    os.makedirs(output_dir, exist_ok=True)
    
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.h5'):
            input_h5_path = os.path.join(input_dir, file_name)
            output_h5_path = os.path.join(output_dir, file_name)
            
            print(f'Processing file: {input_h5_path}')
            process_and_save_h5(model, input_h5_path, output_h5_path, device)

    print('All files processed successfully.')
