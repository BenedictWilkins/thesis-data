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
from functools import partial

from gym.envs.atari import AtariEnv

from .dataset import GymDatasetWriter
from .utils import Log

ROOT_PATH = pathlib.Path("~/.data/")
ROOT_PATH.mkdir(parents=True, exist_ok=True)

def resolve_policy_class(policy):
    from importlib import import_module
    try:
        module_path, class_name = policy.rsplit('.', 1)
        module = import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(policy)

def resolve_path(args):
    args.path = args.path.replace("{env_id}", args.env_id)
    args.path = args.path.replace("{policy}", args.policy.split(".")[-1])
    args.path = pathlib.Path(ROOT_PATH, args.path).expanduser().resolve()
    return args.path

parser = argparse.ArgumentParser(prog="thesis-data", description="")

parser.add_argument("--env_id", "-e", type=str, required=True, help="ID of the gym enviroment.")
parser.add_argument("--path", "-p", type=str, default="{env_id}/{policy}", help=f"Path to save data. Relative paths will be added to {ROOT_PATH}")
parser.add_argument("--policy", "-b", type=str, default="gymu.policy.Uniform", help="Fully qualified class path of the policy to use. Defaults to a uniform policy.")
parser.add_argument("--num_episodes", "-n", type=int, default=1, help="Number of episodes to generate.")
parser.add_argument("--max_episode_length", "-l", type=int, default=10000, help="Maximum episode length.")
parser.add_argument("--mode", "-m", type=str, default="sard", help=", see gymu.mode")
parser.add_argument("--append", "-a", default=False, action='store_true', help="Whether to append episodes to an already existing directory.")
parser.add_argument("--kwargs", "-k", default={}, type=json.loads, help="Additional arguments for the environment, given as dictionary string e.g. \"{'a':1}\"")
args = parser.parse_args()

write_mode = 'a' if args.append else 'w'

env = gymu.make(args.env_id, **args.kwargs)

if isinstance(env.unwrapped, AtariEnv):
    if  "ALE/" not in args.env_id:
        env_id = f"ALE/{args.env_id.split('-')[0]}-v5"
        Log.warning(f"Upgrading environment {args.env_id} to more recent version {env_id} with proper configuration.", stacklevel=2)
        args.env_id = env_id

    kwargs = dict(obs_type='rgb', frameskip=1, mode=0, difficulty=0, repeat_action_probability=0, full_action_space=False, render_mode=None)
    kwargs.update(args.kwargs)
    env = gymu.make(args.env_id, **kwargs)

policy_class = resolve_policy_class(args.policy)
policy = policy_class(env)

path = resolve_path(args)

mode = gymu.mode.mode(args.mode)
iterator = gymu.iterator(env, policy=policy, mode=mode, max_length=args.max_episode_length)
writer = GymDatasetWriter(path, iterator, write_mode=write_mode)
writer.write(args.num_episodes)
writer.write_config()