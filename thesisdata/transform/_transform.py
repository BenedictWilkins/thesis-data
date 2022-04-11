#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Various image transforms. All assume pytorch tensors [...,H,W] [0-1] float32.
   Used for introducing artificial anomalies into trajectories.

   Created on 25-03-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import random
from cv2 import transform
import numpy as np
from typing import Tuple, List, Dict, Union

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F



__all__ = ("RandomMask", "RandomApply", "ApplyWithProbability", "RandomReplaceAction", "RandomResizedPad", "RandomBrightness", "SaltAndPepper", "RandomSplitFlipVertical", "RandomSplitFlipHorizontal",
         # from torchvision.transforms...   
         "GaussianBlur", "RandomResizedCrop", "RandomAffine", "Normalize")

from torchvision.transforms import GaussianBlur, RandomResizedCrop, RandomAffine, Normalize

def _uniform(x, y):
    return (y - x) * torch.rand(1) + x

class ApplyWithProbability(nn.Module):

    def __init__(self, transform : nn.Module, prob : float = 0.1, inplace : bool = False):
        super().__init__()
        self.transform = transform
        self.prob = prob
        self.inplace = inplace
    
    def forward(self, x):
        if torch.rand((1,)).item() > self.prob:
            return x, torch.zeros((1,))
        else:
            return self.transform(x), torch.ones((1,))

class RandomApply(nn.Module):

    def __init__(self, transform : nn.Module, prop : float = 0.1, inplace : bool = False):
        """ Randomly apply the given transformation to a proportion of the given data, the transform is applied individually to each randomly selected element.

        Args:
            transform (nn.Module): transform to apply.
            prop (float, optional): proportion of images to apply to. Defaults to 0.1.
            inplace (bool, optional): whether to apply the transformations in place or not. Defaults to False.
        """
        super().__init__()
        self.prop = prop
        self.transform = transform
        self.inplace = inplace
        
    def forward(self, x : torch.Tensor):
        """ A collection of data [N,D,...]

        Args:
            x (torch.Tensor): data to apply transform.

        Returns:
            torch.Tensor: transformed data.
        """
        if not self.inplace:
            x = x.clone() 
        if len(x.shape) == 1:
            x = x.unsqueeze(1)

        index = torch.randperm(x.shape[0])[:int(self.prop * x.shape[0])]
        for i in index:
            x[i] = self.transform(x[i])
        label = torch.zeros(x.shape[0], dtype=torch.bool)
        label[index] = True
        return x, label

class RandomReplaceAction(nn.Module):
    
    def __init__(self, num_actions):
        super().__init__()
        self.num_actions = num_actions
        
    def forward(self, x):
        x_shape = x.shape
        x = x.view(-1, 1)
        actions = torch.arange(0, self.num_actions).unsqueeze(0)
        indx = actions != x
        actions = actions.repeat(x.shape[0], 1)[indx].view(x.shape[0], self.num_actions-1)
        choice = torch.randint(0, self.num_actions-1, size=x.shape)
        return actions.gather(-1,choice).view(x_shape)

class RandomMask(nn.Module):

    def __init__(self, 
                x : Tuple[float, float] = (0,1),
                y : Tuple[float, float] = (0,1),
                w : Tuple[float, float] = (0.2,0.4),
                h : Tuple[float, float] = (0.2,0.4),
                fill : Union[float, List[float]] = 0,
                inplace : bool = False):
        """ Places a random rectangular mask over part of an image. May be used with TorchScript.

        Args:
            x (Tuple[float, float], optional): x range for the top left corner of the mask [0,1]. Defaults to (0,1).
            y (Tuple[float, float], optional): y range for the top left corner of the mask [0,1]. Defaults to (0,1).
            w (Tuple[float, float], optional): width range for the mask [0,1]. Defaults to (0.2,0.4).
            h (Tuple[float, float], optional): height range for the mask [0,1]. Defaults to (0.2,0.4).
            fill (Union[float, List[float]], optional): fill value. Defaults to 0.
            inplace (bool, optional): transform the image inplace or make a copy. Defaults to False.
        """
        super().__init__()

        self.x, self.y = torch.clip(torch.FloatTensor(x), 0, 1), torch.clip(torch.FloatTensor(y), 0, 1)
        self.w, self.h = torch.clip(torch.FloatTensor(w), 0, 1), torch.clip(torch.FloatTensor(h), 0, 1)
        self.fill = fill
        self.inplace = inplace
        
    def forward(self, img):
        if not self.inplace:
            img = img.clone()
        img_h, img_w = img.shape[-2:]
        w, h = _uniform(self.w[0], self.w[1]), _uniform(self.h[0], self.h[1])
        x1, y1 = _uniform(self.x[0], self.x[1] - w), _uniform(self.y[0], self.y[1] - h)
        x2, y2 = x1 + w, y1 + h
        #x1, y1, x2, y2 = torch.clip(torch.cat([x1,y1,x2,y2]),0,1)
        x1, x2 = int(x1 * img_w), int(x2 * img_w) # CHW
        y1, y2 = int(y1 * img_h), int(y2 * img_h) 
        img[...,y1:y2,x1:x2] = self.fill
        return img
    
 
class RandomResizedPad(nn.Module):
  
    def __init__(self, padding_range : Tuple[int,int] = (5,18), padding_modes : List[str] = ['constant', 'edge', 'reflect', 'symmetric']):
        """
        Randomly pad the edges of an image then resize to match the original input tensor.
        
        Args:
            padding_range (Tuple[int,int], optional): min and max padding values. Defaults to (5,18).
            padding_modes (List[str], optional): padding modes to choose (randomly) from. Defaults to ['constant', 'edge', 'reflect', 'symmetric'].
        """
        super().__init__()
        self.padding_range = padding_range
        self.padding_modes = [*padding_modes]
    
    def forward(self, img):
        padding = random.randint(*self.padding_range)
        #print(padding)
        mode = self.padding_modes[random.randint(0,len(self.padding_modes)-1)]
        _img = F.pad(img, padding, 0, mode)
        return F.resize(_img, img.shape[-2:])

class RandomBrightness(nn.Module):
    def __init__(self, brightness : Tuple[float, float]=(0.2,1.2)):
        """
            Randomly adjust image brightness.

        Args:
            brightness (Tuple[float, float], optional): min and max brightness factor, should be postive. < 1 to decrease, > 1 to increase. Defaults to (0.2,1.2).
        """
        super().__init__()
        self.brightness = (max(0, brightness[0]), brightness[1])
    
    def forward(self, img):
        return torch.clip(img * random.uniform(*self.brightness), 0, 1)

class SaltAndPepper(nn.Module):
    
    def __init__(self, threshold=0.05, pepper=0.1, salt=0.9, mode='snp', inplace=True):
        """
            Adds Salt-and-Pepper noise (https://en.wikipedia.org/wiki/Salt-and-pepper_noise) to an image. 
        Args:
            threshold (float, optional): threshold to introduce salt or 1-threshold for pepper. Defaults to 0.05.
            pepper (float, optional): pepper value. Defaults to 0.1.
            salt (float, optional): salt value. Defaults to 0.9.
            mode (str, optional): mode 'snp' or 'rgb'. 'snp' will apply to all channels, 'rgb' will apply to each channel independantly. For greyscale images they are equivalent. Defaults to 'snp'.
            inplace (bool, optional): whether to add noise inplace or make a copy.
        """
        super().__init__()
        self.threshold = threshold
        self.salt, self.pepper = max(0, min(salt,1)), max(0, min(pepper, 1))
        self.mode = mode
        self._get_image_shape = None
        self.inplace = inplace
        
    def forward(self, img):
        if not self.inplace:
            img = img.clone()
        index = self._get_index(img.shape[-3:]) # [...,C,H,W]
        img[...,index>=1-self.threshold] = self.salt
        img[...,index<=self.threshold] = self.pepper
        return img
        
    def _snp_index(self, shape):
        return torch.rand((1, *shape[1:])).repeat(shape[0], 1, 1) # C,H,W
    
    def _rgb_index(self, shape):
        return torch.rand(shape)

    @property
    def mode(self):
        return self._mode
    
    @mode.setter
    def mode(self, value):
        if value == 'snp':
            self._get_index = self._snp_index
            self._mode = value
        elif value == 'rgb':
            self._get_index = self._rgb_index
        else:
            raise ValueError(f"Invalid mode: {value}, must be 'snp' or 'rgb'")


class RandomSplitFlipVertical(nn.Module):
    
    def __init__(self, split : Union[float, Tuple[float, float]] = 0.5, inplace : bool = False):
        """ Randomly split the image, flip and replace the other side of the image.

        Args:
            split (Union[float, Tuple[float, float]], optional): where to split the image. Defaults to 0.5.
            inplace (bool, optional): whether to transform the image in place. Defaults to False.
        """
        super().__init__()
        self.inplace = inplace
        self.split = split if isinstance(split, tuple) else (split, split) 
    
    def forward(self, img):
        if not self.inplace:
            img = img.clone()
        index = int(img.shape[-2]*random.uniform(*self.split))
        split = img[...,:index,:].flip(-2)
        img[...,-index:,:] = split
        return img

class RandomSplitFlipHorizontal(nn.Module):
    
    def __init__(self, split : Union[float, Tuple[float, float]]  = 0.5, inplace : bool = False):
        """ Randomly split the image, flip and replace the other side of the image.

        Args:
            split (Union[float, Tuple[float, float]], optional): where to split the image. Defaults to 0.5.
            inplace (bool, optional): whether to transform the image in place. Defaults to False.
        """
        super().__init__()
        self.inplace = inplace
        self.split = split if isinstance(split, tuple) else (split, split) 
    
    
    def forward(self, img):
        if not self.inplace:
            img = img.clone()
        index = int(img.shape[-2]*random.uniform(*self.split))
        split = img[...,:index].flip(-1)
        img[...,-index:] = split
        return img
    
    

