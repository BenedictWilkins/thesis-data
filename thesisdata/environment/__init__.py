#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 16-03-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

from gym.envs import register

def register_entry_point(): # entry point hook for openai gym
   register(id="MNIST-v0", entry_point="thesisdata.environment.mnist:MNISTEnvironment")

