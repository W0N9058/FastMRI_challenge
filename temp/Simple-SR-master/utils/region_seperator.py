import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_flat_mask(img, kernel_size=7, std_thresh=0.03, scale=1):
#     print(f"Original image shape: {img.size()}")  # 원본 이미지 형상 출력
    img = F.interpolate(img, scale_factor=scale, mode='bicubic', align_corners=False)
#     print(f"Image shape after interpolation: {img.size()}")  # 보간 후 이미지 형상 출력

#     B, _, H, W = img.size()
    B, C, H, W = img.size()
    
#     r, g, b = torch.unbind(img, dim=1)
#     l_img = (0.2989 * r + 0.587 * g + 0.114 * b).unsqueeze(dim=1)
    
    
    # 디버깅용: 이미지의 형상 확인
#     print(f"Image shape after interpolation: {img.size()}")
    
    # RGB 이미지가 아닌 단일 채널 이미지를 처리하는 부분
    if C == 3:
        r, g, b = torch.unbind(img, dim=1)
        l_img = (0.2989 * r + 0.587 * g + 0.114 * b).unsqueeze(dim=1)
    elif C == 1:
        l_img = img  # 단일 채널 이미지 그대로 사용
    else:
        raise ValueError("Unexpected number of channels: {}".format(C))
    
    
    l_img_pad = F.pad(l_img, (kernel_size//2, kernel_size//2, kernel_size//2, kernel_size//2), mode='reflect')
    unf_img = F.unfold(l_img_pad, kernel_size=kernel_size, padding=0, stride=1)
    std_map = torch.std(unf_img, dim=1, keepdim=True).view(B, 1, H, W)
    mask = torch.lt(std_map, std_thresh).float()

    return mask
