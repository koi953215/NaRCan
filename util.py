import os
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


def get_mgrid(sidelen, vmin=-1, vmax=1):
    if type(vmin) is not list:
        vmin = [vmin for _ in range(len(sidelen))]
    if type(vmax) is not list:
        vmax = [vmax for _ in range(len(sidelen))]
    tensors = tuple([torch.linspace(vmin[i], vmax[i], steps=sidelen[i]) for i in range(len(sidelen))])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, len(sidelen))
    return mgrid

def apply_homography(x, h):
    h = torch.cat([h, torch.ones_like(h[:, [0]])], -1)
    h = h.view(-1, 3, 3)
    x = torch.cat([x, torch.ones_like(x[:, 0]).unsqueeze(-1)], -1).unsqueeze(-1)
    o = torch.bmm(h, x).squeeze(-1)
    o = o[:, :-1] / o[:, [-1]]
    return o

def jacobian(y, x):
    B, N = y.shape
    jacobian = list()
    for i in range(N):
        v = torch.zeros_like(y)
        v[:, i] = 1.
        dy_i_dx = torch.autograd.grad(y,
                                      x,
                                      grad_outputs=v,
                                      retain_graph=True,
                                      create_graph=True)[0]  # shape [B, N]
        jacobian.append(dy_i_dx)
    jacobian = torch.stack(jacobian, dim=1).requires_grad_()
    return jacobian

def overlap_mix(img1, img2, img_order, overlap_num):
    w1 = np.linspace(0, 1, overlap_num)[::-1]
    w2 = 1 - w1
    return w1[img_order] * img1 + w2[img_order] * img2


class VideoFitting(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()

        self.path = path
        if transform is None:
            self.transform = ToTensor()
        else:
            self.transform = transform

        self.video = self.get_video_tensor()
        self.num_frames, _, self.H, self.W = self.video.size()
        self.pixels = self.video.permute(2, 3, 0, 1).contiguous().view(-1, 3)
        self.coords = get_mgrid([self.H, self.W, self.num_frames])

        shuffle = torch.randperm(len(self.pixels))
        self.pixels = self.pixels[shuffle]
        self.coords = self.coords[shuffle]

    def get_video_tensor(self):
        frames = sorted(os.listdir(self.path))
        video = []
        for i in range(len(frames)):
            img = Image.open(os.path.join(self.path, frames[i]))
            img = self.transform(img)
            video.append(img)
        return torch.stack(video, 0)

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels
    

class TestVideoFitting(Dataset):
    def __init__(self, path, transform=None):
        super().__init__()

        self.path = path
        if transform is None:
            self.transform = ToTensor()
        else:
            self.transform = transform

        self.video = self.get_video_tensor()
        self.num_frames, _, self.H, self.W = self.video.size()
        self.pixels = self.video.permute(2, 3, 0, 1).contiguous().view(-1, 3)
        self.coords = get_mgrid([self.H, self.W, self.num_frames])

    def get_video_tensor(self):
        frames = sorted(os.listdir(self.path))
        video = []
        for i in range(len(frames)):
            img = Image.open(os.path.join(self.path, frames[i]))
            img = self.transform(img)
            video.append(img)
        return torch.stack(video, 0)

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels
    

class GroupVideoFitting(Dataset):
    def __init__(self, path, mask_path, transform=None, mask_transform=None):
        super().__init__()

        self.path = path
        self.mask_path = mask_path
        if transform is None:
            self.transform = ToTensor()
        else:
            self.transform = transform
        
        if mask_transform is None:
            self.mask_transform = ToTensor()
        else:
            self.mask_transform = mask_transform

        self.video = self.get_video_tensor()
        self.mask = self.get_mask_tensor()
        self.num_frames, _, self.H, self.W = self.video.size()
        self.pixels = self.video.permute(2, 3, 0, 1).contiguous().view(-1, 3)
        self.mask_pixels = self.mask.permute(2, 3, 0, 1).contiguous().view(-1, 1)
        self.coords = get_mgrid([self.H, self.W, self.num_frames])

        shuffle = torch.randperm(len(self.pixels))
        self.pixels = self.pixels[shuffle]
        self.coords = self.coords[shuffle]
        self.mask_pixels = self.mask_pixels[shuffle]

    def get_video_tensor(self):
        frames = sorted(os.listdir(self.path))
        video = []
        for i in range(len(frames)):
            img = Image.open(os.path.join(self.path, frames[i]))
            img = self.transform(img)
            video.append(img)
        return torch.stack(video, 0)
    
    def get_mask_tensor(self):
        masks = sorted(os.listdir(self.mask_path))
        all_mask = []
        for i in range(len(masks)):
            mask = Image.open(os.path.join(self.mask_path, masks[i]))
            mask = self.mask_transform(mask)
            all_mask.append(mask)
        return torch.stack(all_mask, 0)

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels, self.mask_pixels
    

class TestGroupVideoFitting(Dataset):
    def __init__(self, path, mask_path, back_mask_path, transform=None, mask_transform=None):
        super().__init__()

        self.path = path
        self.mask_path = mask_path
        self.back_mask_path = back_mask_path
        if transform is None:
            self.transform = ToTensor()
        else:
            self.transform = transform
        
        if mask_transform is None:
            self.mask_transform = ToTensor()
        else:
            self.mask_transform = mask_transform

        self.video = self.get_video_tensor()
        self.mask = self.get_mask_tensor()
        self.back_mask = self.get_back_mask_tensor()
        self.num_frames, _, self.H, self.W = self.video.size()
        self.pixels = self.video.permute(2, 3, 0, 1).contiguous().view(-1, 3)
        self.mask_pixels = self.mask.permute(2, 3, 0, 1).contiguous().view(-1, 1)
        self.back_mask_pixels = self.back_mask.permute(2, 3, 0, 1).contiguous().view(-1, 1)
        self.coords = get_mgrid([self.H, self.W, self.num_frames])

    def get_video_tensor(self):
        frames = sorted(os.listdir(self.path))
        video = []
        for i in range(len(frames)):
            img = Image.open(os.path.join(self.path, frames[i]))
            img = self.transform(img)
            video.append(img)
        return torch.stack(video, 0)
    
    def get_mask_tensor(self):
        masks = sorted(os.listdir(self.mask_path))
        all_mask = []
        for i in range(len(masks)):
            mask = Image.open(os.path.join(self.mask_path, masks[i]))
            mask = self.mask_transform(mask)
            all_mask.append(mask)
        return torch.stack(all_mask, 0)
    
    def get_back_mask_tensor(self):
        masks = sorted(os.listdir(self.back_mask_path))
        all_mask = []
        for i in range(len(masks)):
            mask = Image.open(os.path.join(self.back_mask_path, masks[i]))
            mask = self.mask_transform(mask)
            all_mask.append(mask)
        return torch.stack(all_mask, 0)

    def __len__(self):
        return 1

    def __getitem__(self, idx):    
        if idx > 0: raise IndexError
            
        return self.coords, self.pixels, self.mask_pixels, self.back_mask_pixels