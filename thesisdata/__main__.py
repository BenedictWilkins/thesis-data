#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 16-03-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import argparse
import pathlib
import gymu
import json
import numpy as np

from .utils import getLogger, resolve_class, GymDatasetWriter
Logger = getLogger()

ROOT_PATH = pathlib.Path("~/.data/").expanduser().resolve()
ROOT_PATH.mkdir(parents=True, exist_ok=True)

def resolve_path(args, path):
    path = path.replace("{env_id}", args.env_id)
    path = path.replace("{policy}", args.policy.split(".")[-1])
    path = pathlib.Path(ROOT_PATH, path).expanduser().resolve()
    return path

parser = argparse.ArgumentParser(prog="thesis-data", description="")

parser.add_argument("--env_id", "-e", type=str, required=True, help="ID of the gym enviroment.")
parser.add_argument("--path", "-p", type=str, default="{env_id}/{policy}", help=f"Path to save data. Relative paths will be added to {ROOT_PATH}")
parser.add_argument("--policy", "-b", type=str, default="gymu.policy.Uniform", help="Fully qualified class path of the policy to use. Defaults to a uniform policy.")
parser.add_argument("--num_episodes", "-n", type=int, default=1, help="Number of episodes to generate.")
parser.add_argument("--max_episode_length", "-l", type=int, default=10000, help="Maximum episode length.")
# TODO parser.add_argument("--min_episode_length", "-k", type=int, default=100, help="Discard an episode if its below this length.")
parser.add_argument("--mode", "-m", type=str, default="sardi", help=", see gymu.mode")
parser.add_argument("--append", "-a", default=False, action='store_true', help="Whether to append episodes to an already existing directory.")
parser.add_argument("--env_kwargs", "-k", default={}, type=json.loads, help="Additional arguments for the environment, given as dictionary string e.g. \"{'a':1}\"")

parser.add_argument("--train", default=False, action='store_true', help="Generate training data, append 'train' to path.")
parser.add_argument("--validate", default=False, action='store_true', help="Generate validation data, append 'validate' to path.")
parser.add_argument("--test", default=False, action='store_true', help="Generate testing data, append 'test' to path.")

args = parser.parse_args()

path_append_index = np.array([args.train, args.validate, args.test])
if path_append_index.sum() == 1:
    path_append = np.array(['train', 'validate', 'test'])[path_append_index].item()
    args.path = str(pathlib.PurePath(args.path, path_append))
elif path_append_index.sum() > 1:
    raise ValueError("Only one of '--train', '--validate', '--test' may be specified at a time.")

write_mode = 'a' if args.append else 'w'

if "stable_baselines3" in args.policy:
    from .environment import sb3
    env, policy = sb3.load(args)
else: 
    env = gymu.make(args.env_id, **args.env_kwargs)
    policy = resolve_class(args.policy)(env)

path = resolve_path(args, args.path)
Logger.info(f"Dataset path: {path}")

mode = gymu.mode.mode(args.mode)

if gymu.intercept.interceptable(env):
    iterator = gymu.intercept.InterceptIterator(env, policy=policy, mode=mode, max_length=args.max_episode_length)
else:
    iterator = gymu.iter.Iterator(env, policy=policy, mode=mode, max_length=args.max_episode_length)

writer = GymDatasetWriter(path, iterator, write_mode=write_mode)
writer.write(args.num_episodes)
writer.write_config()



















#if isinstance(env.unwrapped, AtariEnv):
#    if  "ALE/" not in args.env_id:
#        env_id = f"ALE/{args.env_id.split('-')[0]}-v5"
#        Log.warning(f"Upgrading environment {args.env_id} to more recent version {env_id} with proper configuration.", stacklevel=2)
#        args.env_id = env_id
#
#    kwargs = dict(obs_type='rgb', frameskip=1, mode=0, difficulty=0, repeat_action_probability=0, full_action_space=False, render_mode=None)
#    kwargs.update(args.kwargs)
#    env = gymu.make(args.env_id, **kwargs)