#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 16-03-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import numpy as np
import torch
import torchvision
import itertools
import pathlib
import gym

__all__ = ("MNISTEnvironment", "MNIST_PATH")

MNIST_PATH = pathlib.Path(__file__)
MNIST_PATH = next(p for p in MNIST_PATH.parents if p.name == "thesisdata").parent
MNIST_PATH = pathlib.Path(MNIST_PATH, "data")

class MNISTData:    

    __INSTANCE__ = None

    def __new__(cls):
        if MNISTData.__INSTANCE__ is None:
            __INSTANCE__ = super().__new__(cls)
            __INSTANCE__.dataset = torchvision.datasets.MNIST(str(MNIST_PATH), transform=torchvision.transforms.ToTensor(), download=True)
        return __INSTANCE__

class MNISTEnvironment(gym.Env):

    def __init__(self, num_actions=2):
        super().__init__()
        dataset = MNISTData().dataset
        x, y = dataset.data.float() / 255., dataset.targets

        self._groups = [y == i for i in range(y.max() + 1)]
        self._groups = [(x[i], y[i]) for i in self._groups]
        self._groups = [x.unsqueeze(1) for x,_ in self._groups]
        
        self.action_space = gym.spaces.Discrete(num_actions)
        self.observation_space = gym.spaces.Box(0,1,shape=x.shape[1:])
        
        self._index = 0
        self._random = np.random.default_rng()
        self._current_group = self._random.integers(0, self.action_space.n)
        
    def get_action_meanings(self):
        return [f"(x+{i + 1}) % {self.action_space.n}" for i in range(self.action_space.n)]

    def step(self, action):
        assert action in self.action_space
        self._current_group = (self._current_group + action + 1) % (len(self._groups)) 
        group = self._groups[self._current_group]
        self._index += self._random.integers(1, 20) # 20 gives some randomness to the transitions...
        self._index = min(group.shape[0] - 1, self._index)
        done = self._index >= group.shape[0] - 1
        state = group[self._index]
        reward = 0. # TODO what kind of reward do we want? 
        return state, reward, done, None

    def reset(self): # random starting index
        self._index = 0
        self._current_group = self._random.integers(0, self.action_space.n)
        return self._groups[self._current_group][self._index]




