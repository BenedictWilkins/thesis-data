#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Uses pre-trained models from stable baselines, requires loading the environment is a specific way (with correct wrappers etc). This code has been taken from stable_baselines3 zoo: https://github.com/DLR-RM/rl-baselines3-zoo

   Note that this requires v0.21 of gym, it breaks with the newer atari-py version (which is very annoying -_-).
   
   Created on 21-03-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import pathlib
import glob
import sys
import os
import yaml
import numpy as np
import gymu
import gym

from .sb3_zoo_utils import ALGOS, create_test_env, get_saved_hyperparams

# it doesnt like relative imports here??? wtf.
from thesisdata.utils._logging import getLogger
Logger = getLogger()

from thesisdata.utils import get_project_root_directory, resolve_class

DEFAULT_MODEL_PATH = pathlib.Path(__file__).parent
DEFAULT_MODEL_PATH = pathlib.Path(DEFAULT_MODEL_PATH, "rl-trained-agents/")

class SB3PolicyWrapper: # wrap the s3b policy so that randomness may be introduced & works with gymu iterators.

   def __init__(self, policy, action_space, eps=0, deterministic=False):
      self.policy = policy
      self.eps = eps
      self.action_space = action_space
      self.deterministic = deterministic
      self.hidden_state = None

   def __call__(self, state):
      # TODO recurrent policies not supported... reset is a bit tricky...
      if np.random.uniform() > self.eps:
         action, self.hidden_state = self.policy.predict(state, deterministic=self.deterministic)
      else:
         action = self.action_space.sample() # TODO this doesnt work out of the box... which is annoying! maybe create an issue or something?
         action = np.array([action])
      return action

class InfoWrapper(gym.Wrapper): # get atari envs to conform to gym API.
   def step(self, a):
      state, reward, done, info = self.env.step(a)
      return state, reward, done, info[0]
   def reset(self):
      return self.env.reset(), {}
      
def load(args):
   # use pretrained models...
   path = pathlib.Path(DEFAULT_MODEL_PATH, args.policy.split(".")[-1].lower(), f"{args.env_id}_1")
   env_id = args.env_id
   env_kwargs = args.env_kwargs
   Logger.info(f"Using SB3 path: {path}")
   
   # LOAD ENVIRONMENT
   stats_path = os.path.join(path, env_id)
   hyperparams, stats_path = get_saved_hyperparams(stats_path, norm_reward=False, test_mode=True)

   # load env_kwargs if existing
   args_path = os.path.join(path, env_id, "args.yml")
   if os.path.isfile(args_path):
      with open(args_path, "r") as f:
         loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
         if loaded_args["env_kwargs"] is not None:
               env_kwargs = loaded_args["env_kwargs"]
   # overwrite with command line arguments
   if env_kwargs is not None:
      env_kwargs.update(env_kwargs)

   env = create_test_env(
         env_id,
         n_envs=1,
         stats_path=stats_path,
         seed=args.__dict__.get('seed', np.random.randint(100000)),
         log_dir=None, # logdir?
         should_render=False,
         hyperparams=hyperparams,
         env_kwargs=env_kwargs,
      )
   env = InfoWrapper(env)

   # LOAD POLICY
   policy_cls = resolve_class(args.policy)
   
   checkpoints = glob.glob(str(pathlib.PurePath(path, "**/*.zip")), recursive=True)
   checkpoint_path = checkpoints[-1] # there should only be one...
   Logger.info(f"Using policy checkpoint: {checkpoint_path}")
   policy_kwargs = {}
   policy_kwargs['buffer_size'] = 1 # dont need a buffer...

   newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8
   custom_objects = {}
   if newer_python_version:
      custom_objects = {
         "learning_rate": 0.0,
         "lr_schedule": lambda _: 0.0,
         "clip_range": lambda _: 0.0,
      }

   policy = policy_cls.load(checkpoint_path, env, custom_objects=custom_objects, **policy_kwargs)
   policy = SB3PolicyWrapper(policy, env.action_space, args.__dict__.get('policy_eps', 0.1), deterministic=False)
   return env, policy



