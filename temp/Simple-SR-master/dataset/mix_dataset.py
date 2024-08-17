import cv2
import numpy as np
import os
import random

import torch
from torch.utils.data import Dataset

import h5py  # h5py import 추가
import matplotlib.pyplot as plt  # matplotlib import 추가


class MixDataset(Dataset):
    def __init__(self, hr_paths, lr_paths, config, sample_ratio=0.1):
        self.hr_paths = hr_paths
        self.lr_paths = lr_paths
        self.phase = config.PHASE
        self.input_width, self.input_height = config.INPUT_WIDTH, config.INPUT_HEIGHT
        self.scale = config.SCALE
        self.repeat = config.REPEAT
        self.value_range = config.VALUE_RANGE
        
        self.sample_ratio = sample_ratio

        self._load_data()

    def _load_data(self):
        
        # (병현) h5 파일을 이미지로 처리하는 코드 필요할 듯
        
        assert len(self.lr_paths) == len(self.hr_paths), 'Illegal hr-lr dataset mappings.'

        self.hr_list = []
        self.lr_list = []
        
        for hr_path in self.hr_paths:
            hr_files = sorted(os.listdir(hr_path))
#             print(hr_files)

            for hr_file in hr_files:
                if hr_file.endswith('.h5'):
                    with h5py.File(os.path.join(hr_path, hr_file), 'r') as f:
                        hr_images = f['image_label'][:]  # Load HR images from 'image_label' dataset
                        for i in range(hr_images.shape[0]):
                            self.hr_list.append(hr_images[i])

        for lr_path in self.lr_paths:
            lr_files = sorted(os.listdir(lr_path))
#             print(lr_files)
            
            for lr_file in lr_files:
                if lr_file.endswith('.h5'):
                    with h5py.File(os.path.join(lr_path, lr_file), 'r') as f:
                        lr_images = f['reconstruction'][:]  # Load LR images from 'image_grappa' dataset
                        for i in range(lr_images.shape[0]):
                            self.lr_list.append(lr_images[i])

        assert len(self.hr_list) == len(self.lr_list), 'Illegal hr-lr mappings.'
        
        # 이미지 개수를 의미
        self.data_len = len(self.hr_list)
        self.full_len = self.data_len * self.repeat


    def __len__(self):
        return self.full_len

    def __getitem__(self, index):
        # index 값이 데이터 길이보다 클 경우를 대비한 코드
        idx = index % self.data_len

        img_hr = self.hr_list[idx]
        img_lr = self.lr_list[idx]
#         print(f"변경 전: img_hr.shape = {img_hr.shape}")
        
        # 이미지를 (H, W, 1) 형태로 변경
        img_hr = np.expand_dims(img_hr, axis=0)  # (H, W) -> (1, H, W)
        img_lr = np.expand_dims(img_lr, axis=0)  # (H, W) -> (1, H, W)
#         print(f"변경 후: img_hr.shape = {img_hr.shape}")
        
#         url_hr = self.hr_list[idx]
#         url_lr = self.lr_list[idx]
#         img_hr = cv2.imread(url_hr, cv2.IMREAD_COLOR)
#         img_lr = cv2.imread(url_lr, cv2.IMREAD_COLOR)

        if self.phase == 'train':
            h, w = img_lr.shape[1:3]
            s = self.scale
            
#             # input_height, width가 뭔가 다른걸 의미하는건가?, 코드 보면 작은 단위로 나누는거 같은데 crop이라 그런가
#             # crop인거 같은데 해당 부분 주석 처리 해버리자
#             # random cropping
#             y = random.randint(0, h - self.input_height)
#             x = random.randint(0, w - self.input_width)
            
#             img_lr = img_lr[y: y + self.input_height, x: x + self.input_width, :]
#             img_hr = img_hr[y * s: (y + self.input_height) * s,
#                             x * s: (x + self.input_width) * s, :]

            # horizontal flip
            if random.random() > 0.5:
                img_lr = cv2.flip(img_lr[0], 1).reshape(1, *img_lr.shape[1:])  # (1, H, W)
                img_hr = cv2.flip(img_hr[0], 1).reshape(1, *img_hr.shape[1:])  # (1, H, W)
            # vertical flip
            if random.random() > 0.5:
                img_lr = cv2.flip(img_lr[0], 0).reshape(1, *img_lr.shape[1:])  # (1, H, W)
                img_hr = cv2.flip(img_hr[0], 0).reshape(1, *img_hr.shape[1:])  # (1, H, W)
            # rotation 90 degree
            if random.random() > 0.5:
                img_lr = np.transpose(img_lr, (0, 2, 1))  # (1, H, W) -> (1, W, H)
                img_hr = np.transpose(img_hr, (0, 2, 1))  # (1, H, W) -> (1, W, H)
        
#         # 이미지 RBG 바꾸는 부분
#         # HWC는 각각 Height, Width, Channel를 의미한다고 함
# #         # BGR to RGB, HWC to CHW, uint8 to float32
# #         img_lr = np.transpose(img_lr[:, :, [2, 1, 0]], (2, 0, 1)).astype(np.float32)
# #         img_hr = np.transpose(img_hr[:, :, [2, 1, 0]], (2, 0, 1)).astype(np.float32)

# #         # 우리 데이터는 모노 이미지이므로 (1, height, width) 형태로 변경하기, (height, width)이 맞다면 expand_dims 제외하기
# #         img_lr = np.expand_dims(img_lr, axis=0).astype(np.float32)  # (H, W) -> (1, H, W)
# #         img_hr = np.expand_dims(img_hr, axis=0).astype(np.float32)  # (H, W) -> (1, H, W)
        
# #         print(img_lr.shape)
# #         print(img_hr.shape)
        
        # 범위를 0-1로 보정하는건데 우리 코드는 0.001인가 그럼 내일 한번 확인해보기
        # numpy array to tensor, [0, 255] to [0, 1]
#         img_lr = torch.from_numpy(img_lr).float() / self.value_range
#         img_hr = torch.from_numpy(img_hr).float() / self.value_range
        img_lr = torch.from_numpy(img_lr).float() * 1024.0
        img_hr = torch.from_numpy(img_hr).float() * 1024.0
        
#         print("lr_img pixel max:", torch.max(img_lr))

        return img_lr, img_hr


if __name__ == '__main__':
#     from easydict import EasyDict as edict
#     config = edict()
#     config.PHASE = 'train'
#     config.INPUT_WIDTH = config.INPUT_HEIGHT = 64
#     config.SCALE = 4
#     config.REPEAT = 1
#     config.VALUE_RANGE = 255.0

# #     D = MixDataset(hr_paths=['/data/liwenbo/datasets/DIV2K/DIV2K_train_HR_sub'],
# #                    lr_paths=['/data/liwenbo/datasets/DIV2K/DIV2K_train_LR_bicubic_sub/X4'],
# #                    config=config)
#     D = MixDataset(hr_paths=['/home/Data/train/image'],
#                    lr_paths=['/home/Data/train/image'],
#                    config=config)
#     print(D.data_len, D.full_len)
#     lr, hr = D.__getitem__(5)
#     print(lr.size(), hr.size())
#     print('Done')

######################### 테스트용 #########################
    from easydict import EasyDict as edict
    config = edict()
    config.PHASE = 'train'
    config.INPUT_WIDTH = config.INPUT_HEIGHT = 384
    config.SCALE = 1
    config.REPEAT = 1
    config.VALUE_RANGE = 0.001

    D = MixDataset(hr_paths=['/home/Data/train/image'],
                   lr_paths=['/home/Data/train/image'],
                   config=config)
    
    print(f"Data length: {D.data_len}")
    print(f"Full length (data length * repeat): {D.full_len}")

    lr, hr = D.__getitem__(5)
    
    print(f"Low Resolution Image Size: {lr.size()}")
    print(f"High Resolution Image Size: {hr.size()}")

#     # 이미지를 시각화
#     lr_np = lr.numpy().squeeze() * config.VALUE_RANGE
#     hr_np = hr.numpy().squeeze() * config.VALUE_RANGE

#     plt.figure(figsize=(10, 5))
#     plt.subplot(1, 2, 1)
#     plt.title('Low Resolution')
#     plt.imshow(lr_np, cmap='gray')
#     plt.subplot(1, 2, 2)
#     plt.title('High Resolution')
#     plt.imshow(hr_np, cmap='gray')
#     plt.show()

    print('Done')