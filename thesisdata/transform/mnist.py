#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 25-03-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as F

from . import _transform

class MNISTTransform:
    """ 
        Simple transform for MNIST images to get it into a usable format.: (1,H,W) [0-1] float32 
    """
    
    def __init__(self, device="cuda:0"):
        super().__init__()
        self.device = device
    
    def __call__(self, x):
        x = x.unsqueeze(1).to(self.device).float()
        x /= 255.
        return x

RandomApply = _transform.RandomApply

# MNIST ANOMALY TRANSFORMS
RandomResizedPad            = lambda padding_range=(5,13): _transform.RandomResizedPad(padding_range=padding_range, padding_modes=['reflect', 'symmetric'])
RandomResizedCrop           = lambda: T.RandomResizedCrop((28,28), interpolation=T.InterpolationMode.NEAREST)
RandomAffine                = lambda angle=90, translate=(0,0.3), scale=(0.5,1.5): T.RandomAffine(angle, translate, scale)
RandomBrightness            = lambda brightness=(0.2,0.5): _transform.RandomBrightness(brightness=brightness)
GaussianBlur                = lambda kernel_size=9: T.GaussianBlur(kernel_size=kernel_size)
SaltAndPepper               = lambda threshold=0.15, pepper=0.1, salt=0.9, inplace=False: _transform.SaltAndPepper(threshold=threshold, pepper=pepper, salt=salt, mode='rgb', inplace=inplace)
RandomSplitFlipVertical     = lambda split=(0.3,0.6), inplace=False: _transform.RandomSplitFlipVertical(split=split, inplace=inplace)
RandomSplitFlipHorizontal   = lambda split=(0.3,0.6), inplace=False: _transform.RandomSplitFlipHorizontal(split=split, inplace=inplace)
RandomMask                  = lambda w=(1/5,1/2), h=(1/5,1/2), inplace=False: _transform.RandomMask(x=(0.2,0.8), y=(0.2,0.8), w=w, h=h, fill=0, inplace=inplace)

transforms = [
    RandomResizedPad,          
    RandomResizedCrop,           
    RandomAffine,               
    RandomBrightness,            
    GaussianBlur,                
    SaltAndPepper,               
    RandomSplitFlipVertical,     
    RandomSplitFlipHorizontal,   
    RandomMask,                   
]