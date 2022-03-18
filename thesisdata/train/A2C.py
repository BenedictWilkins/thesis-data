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
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

# Parallel environments
env_id = "Breakout-v4"
env = make_vec_env(env_id, n_envs=64)
load_path = f"models/A2C/{env_id}"

model = A2C("CnnPolicy", env, verbose=1, policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5)))
model.learn(total_timesteps=500000)
model.save(load_path)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
