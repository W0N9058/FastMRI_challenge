import numpy as np
import h5py
import glob
import os
import torch
from utils.common.loss_function import SSIMLoss
import torch.nn.functional as F

class SSIM(SSIMLoss):
    def __init__(self, win_size: int = 7, k1: float = 0.01, k2: float = 0.03):
        super().__init__(win_size, k1, k2)
            
    def forward(self, X, Y, data_range):
        if len(X.shape) != 2:
            raise NotImplementedError('Dimension of first input is {} rather than 2'.format(len(X.shape)))
        if len(Y.shape) != 2:
            raise NotImplementedError('Dimension of first input is {} rather than 2'.format(len(Y.shape)))
            
        X = X.unsqueeze(0).unsqueeze(0)
        Y = Y.unsqueeze(0).unsqueeze(0)
        data_range = torch.tensor([data_range], device=X.device)
        
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2
        ux = F.conv2d(X, self.w)
        uy = F.conv2d(Y, self.w)
        uxx = F.conv2d(X * X, self.w)
        uyy = F.conv2d(Y * Y, self.w)
        uxy = F.conv2d(X * Y, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D
        return S.mean()
    

def calculate_ssim(data_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ssim_calculator = SSIM().to(device=device)
    files = glob.glob(os.path.join(data_path, '*.h5'))
    ssim_total = 0
    count = 0

    for file_path in files:
        with h5py.File(file_path, 'r') as hf:
            if 'image_grappa' in hf and 'image_label' in hf:
                grappa = np.array(hf['image_grappa'])
                label = np.array(hf['image_label'])
                
                # 데이터의 최대 값으로 data_range 설정
                data_range = max(grappa.max(), label.max())

                # SSIM 계산
                for i in range(grappa.shape[0]):
                    grappa_slice = torch.from_numpy(grappa[i]).float().to(device)
                    label_slice = torch.from_numpy(label[i]).float().to(device)
                    ssim_value = ssim_calculator(grappa_slice, label_slice, data_range)
                    ssim_total += ssim_value.item()
                    count += 1
                print(f'File {file_path}, SSIM: {ssim_value.item()}')

    average_ssim = ssim_total / count if count > 0 else 0
    print(f'Average SSIM across all processed files: {average_ssim:.4f}')

if __name__ == '__main__':
    data_path = '/home/Data/train/image'
    calculate_ssim(data_path)