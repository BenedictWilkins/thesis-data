#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 16-03-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import gym

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
env_id = "CartPole-v1"
env = make_vec_env(env_id, n_envs=4)
load_path = f"models/A2C/{env_id}"

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=25000)
model.save(load_path)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
