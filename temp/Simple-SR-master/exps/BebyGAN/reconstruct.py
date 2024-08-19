import os
import sys
import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)) + '/../..')

from utils.common import tensor2img

def process_and_save_h5(model, input_h5_path, output_h5_path, device):
    with torch.no_grad():
        with h5py.File(input_h5_path, 'r') as f:
            recon_data = f['reconstruction'][:]  # (N, 384, 384) 또는 다른 크기
            
            processed_data = []
            
            for i in range(recon_data.shape[0]):
                lr_img = torch.tensor(recon_data[i:i+1, :, :]).unsqueeze(0).to(device)  # Add batch and channel dims
                # print("lr_img max value:", lr_img.max().item())
                # print("lr_img min value:", lr_img.min().item())

                # 여기 과정에서 문제발생
                output = model.G(lr_img)
                
                output_img = tensor2img(output)
                # output_img = output.squeeze(0).cpu().numpy()  # float32로 직접 변환

                print("output_img max value:", output_img.max().item())
                print("output_img min value:", output_img.min().item())
                
                processed_data.append(output_img)
            
            processed_data = np.stack(processed_data, axis=0)
        
        # Save the modified reconstruction data to a new h5 file
        with h5py.File(output_h5_path, 'w') as f:
            f.create_dataset('reconstruction', data=processed_data)
            
        # print(f'Saved processed data to {output_h5_path}')

def process_directory(model, input_dir, output_dir, device):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.h5'):
            input_h5_path = os.path.join(input_dir, file_name)
            output_h5_path = os.path.join(output_dir, file_name)
            
            # print(f'Processing file: {input_h5_path}')
            process_and_save_h5(model, input_h5_path, output_h5_path, device)

if __name__ == '__main__':
    from config import config
    from network import Network
    from utils.model_opr import load_model

    # Model and configuration setup
    config.VAL.DATASETS = ['FASTMRI']
    config.VAL.SAVE_IMG = True

    model = Network(config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.G = model.G.to(device)

    model_path = '/root/FastMRI_challenge/temp/Simple-SR-master/exps/BebyGAN/log/models/5000_G.pth'
    load_model(model.G, model_path, cpu=True)

    # Input and output directories
    input_base_dir = '/root/result/test_Varnet/reconstructions_leaderboard/'
    output_base_dir = '/root/result/test_Varnet/reconstructions_leaderboard_processed/'

    # Process the private and public directories
    for subfolder in ['private', 'public']:
        input_dir = os.path.join(input_base_dir, subfolder)
        output_dir = os.path.join(output_base_dir, subfolder)
        
        process_directory(model, input_dir, output_dir, device)

    print('All files processed successfully.')