import numpy as np
import torch

def to_tensor(data):
    """
    Convert numpy array to PyTorch tensor. For complex arrays, the real and imaginary parts
    are stacked along the last dimension.
    Args:
        data (np.array): Input numpy array
    Returns:
        torch.Tensor: PyTorch version of data
    """
    return torch.from_numpy(data)

class DataTransform:
    def __init__(self, isforward, max_key):
        self.isforward = isforward
        self.max_key = max_key
    def __call__(self, mask, input, target, attrs, fname, slice):
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1
        
        kspace = to_tensor(input * mask)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
        return mask, kspace, target, maximum, fname, slice

"""
import numpy as np
import torch
from utils.mraugment.data_augment import DataAugmentor
from utils.mraugment.data_transforms import VarNetDataTransform

def to_tensor(data):
    return torch.from_numpy(data)

class DataTransform:
    def __init__(self, isforward, max_key, augmentor=None):
=======
>>>>>>> 5e92c47a359dbb44e7325e6a002ad941b6740c54
        self.isforward = isforward
        self.max_key = max_key
    def __call__(self, mask, input, target, attrs, fname, slice):
        if not self.isforward:
            target = to_tensor(target)
            maximum = attrs[self.max_key]
        else:
            target = -1
            maximum = -1
        
        kspace = to_tensor(input * mask)
        kspace = torch.stack((kspace.real, kspace.imag), dim=-1)
        mask = torch.from_numpy(mask.reshape(1, 1, kspace.shape[-2], 1).astype(np.float32)).byte()
<<<<<<< HEAD
        
        if self.augmentor:
            # kspace = self.augmentor(kspace, target.size())
            kspace, target = self.augmentor(kspace, target.size() if isinstance(target, torch.Tensor) else (kspace.shape[-3], kspace.shape[-2]))
        
        
        return mask, kspace, target, maximum, fname, slice

    
def get_augmentor(args, current_epoch_fn):
    return DataAugmentor(args, current_epoch_fn)

"""