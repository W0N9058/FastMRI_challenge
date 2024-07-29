import torch
import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

def load_data(file_path, input_key, target_key, device='cuda'):
    with h5py.File(file_path, 'r') as f:
        masked_data = torch.tensor(np.array(f[input_key]), device=device)
        original_data = torch.tensor(np.array(f[target_key]), device=device)
    return masked_data, original_data

def calculate_deviation(masked, original):
    deviation = torch.abs(original - masked)
    return deviation

def visualize_deviation_histogram(deviation, title='Deviation between Original and Masked Data'):
    deviation_cpu = deviation.cpu().numpy()
    plt.figure(figsize=(10, 6))
    plt.hist(deviation_cpu.flatten(), bins=50, alpha=0.75)
    plt.title(title)
    plt.xlabel('Deviation')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

def analyze_bias(data_dir, input_key, target_key, device='cuda'):
    all_deviation = []

    # 데이터 파일 리스트 가져오기
    data_files = list(Path(data_dir).rglob('*.h5'))
    
    for file_path in tqdm(data_files, desc="Processing files"):
        masked_data, original_data = load_data(file_path, input_key, target_key, device)
        deviation = calculate_deviation(masked_data, original_data)
        all_deviation.append(deviation.cpu().numpy())
    
    # 전체 편차 데이터 결합
    all_deviation = np.concatenate(all_deviation)
    visualize_deviation_histogram(torch.tensor(all_deviation), title="Overall Deviation Histogram")

# 데이터셋 경로와 키 설정
data_dir = 'path/to/your/dataset'
input_key = 'kspace'
target_key = 'image'

analyze_bias(data_dir, input_key, target_key)
