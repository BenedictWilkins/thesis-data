#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 16-03-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import pathlib
import glob
import yaml
import gymu # ensures env seralization works properly.

from tqdm.auto import tqdm

from ..utils import Log

WRITE_MODE_APPEND = 'a'
WRITE_MODE_WRITE = 'w'

__all__ = ("GymDatasetWriter", "get_environment_config")

class GymDatasetWriter:

    def __init__(self, path, iterator, write_mode=WRITE_MODE_APPEND):
        self.path = pathlib.Path(path)

        self.iterator = iterator
        self.num_episodes = 0
        exist_ok = write_mode == WRITE_MODE_APPEND
        self.path.mkdir(parents=True, exist_ok=exist_ok)
        if write_mode == WRITE_MODE_APPEND:
            self.num_episodes = len(list(self.path.iterdir()))

    def write(self, n):
        for _ in range(n):
            path = pathlib.Path(self.path, str(self.num_episodes).zfill(8))
            Log.info(f"Writing episode: {path}")
            gymu.data.write_episode(tqdm(self.iterator), path=path)
            self.num_episodes += 1

    def write_config(self):
        config = dict(**get_environment_config(self.iterator.env),
                        policy = self._get_classname(self.iterator.policy),
                        mode = self.iterator.mode.__name__)
        with pathlib.Path(self.path, "meta.yaml").open('w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def _get_classname(self, obj):
        cls = type(obj)
        module = cls.__module__
        name = cls.__qualname__
        if module is not None and module != "__builtin__":
            name = module + "." + name
        return name

def get_environment_config(env): # TODO this might be moved over to a yaml resolver/constructor... its a little bit tricky to deal with env wrappers...
    """
        Get dictionary that describes environment.
    """
    spec = {k:v for k,v in env.spec.__dict__.items()}
    spec['env_id'] = env.spec.id
    return dict(spec=spec, action_space=env.action_space, observation_space=env.observation_space)
    


