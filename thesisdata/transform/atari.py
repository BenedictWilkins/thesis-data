#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 31-03-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import torch

def SpaceInvaders(x):
    x['state'] /= 0.5
    x['state'] = torch.clip(x['state'], 0, 1)
    return x

dataset_transforms = {
    "SpaceInvadersNoFrameSkip-v0" : SpaceInvaders
}




