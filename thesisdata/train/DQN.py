#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 16-03-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"


import gym

from stable_baselines3 import DQN, a2c
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.atari_wrappers import AtariWrapper

env_id = "ALE/Breakout-v5"
load_path = f"models/DQN/{env_id}"

class FireReset(gym.Wrapper): # ensure the game actually starts
    def reset(self):
        self.env.reset()
        fire = self.env.get_action_meanings().index("FIRE")
        state, *_ = self.env.step(fire)
        return state

def make(env_id, render=False):
    if "ALE" in env_id:
        render_mode = None if not render else 'human'
        env = gym.make(env_id, obs_type='rgb', frameskip=1, mode=0, difficulty=0, repeat_action_probability=0, full_action_space=False, render_mode=render_mode)
        #env = FireReset(env)
        env = AtariWrapper(env)
    else:
        raise ValueError("only ALE environments are supported.")
    return env

def enjoy(env_id, model):
    env = make(env_id, render=True)
    obs = env.reset()
    for i in range(100):
        print(i)
        action, _states = model.predict(obs, deterministic=False)
        obs, rewards, dones, info = env.step(action)

def train(env_id):
    env = make(env_id)
    # Instantiate the agent
    #model = DQN('CnnPolicy', env, buffer_size=100000, learning_starts=10000, verbose=1, device="cuda:0")
    model = A2C()

    for i in range(10):
        model.learn(total_timesteps=10000)
        model.save(load_path + f"/{i}")
        #enjoy(env_id, model)

train(env_id)