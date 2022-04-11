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
import stable_baselines3
import yaml
import gymu # ensures env seralization works properly.
import numpy as np
import gym

from tqdm.auto import tqdm

from ._logging import getLogger
Logger = getLogger()

WRITE_MODE_APPEND = 'a'
WRITE_MODE_WRITE = 'w'

__all__ = ("GymDatasetWriter",)

class GymDatasetWriter:

    def __init__(self, path, iterator, write_mode=WRITE_MODE_APPEND):
        self.path = pathlib.Path(path)

        self.iterator = iterator
        self.num_episodes = 0
        exist_ok = write_mode == WRITE_MODE_APPEND
        self.path.mkdir(parents=True, exist_ok=exist_ok)
        if write_mode == WRITE_MODE_APPEND:
            self.num_episodes = len(list(self.path.iterdir()))

        self._write_wrapped = iterator.env
        self._write_wrappers = []
        
        # this is a bit of hack to get stable baselines in the right save format... for some reason they are working with uint8 HWC format images? why!?
        if isinstance(iterator.env.observation_space, gym.spaces.Box) and len(iterator.env.observation_space.shape) == 3:
            # if the environment has integer image observations (uint8) then wrap it as a float...
            if issubclass(iterator.env.observation_space.dtype.type, np.integer):
                self._write_wrapped = gymu.wrappers.image.Float(self._write_wrapped) # convert to 0-1 float observations for writing...
                self._write_wrappers.append(self._write_wrapped)
            # if the enviroment is in HWC format, then wrap it as CHW format
            if iterator.env.observation_space.shape[-1] in [1,3]: # guess channel dimension...
                self._write_wrapped = gymu.wrappers.image.CHW(self._write_wrapped) # convert to CHW observations for writing...
                self._write_wrappers.append(self._write_wrapped)

    def _write_wrapper_iter(self):
        for x in self.iterator:
            x = dict(x.items())
            for wrapper in self._write_wrappers:
                assert isinstance(wrapper, gym.ObservationWrapper)
                if 'state' in x:
                    x['state'] = wrapper.observation(x['state'])
                if 'nextstate' in x:
                    x['nextstate'] = wrapper.observation(x['nextstate'])
            yield self.iterator.mode(**x)

    def write(self, n):
        for _ in range(n):
            path = pathlib.Path(self.path, str(self.num_episodes).zfill(8))
            Logger.info(f"Writing episode: {path}")
            gymu.data.write_episode(self._write_wrapper_iter(), path=path)
            self.num_episodes += 1

    def write_config(self):
        config = dict(**get_environment_config(self._write_wrapped),
                        policy = self._get_classname(self.iterator.policy),
                        mode = self.iterator.mode.__name__)
        # TODO include wrappers... the easiest thing to do might be just to register the environment under thesis/<env_id> elsewhere ???  hmmm...
        
        with pathlib.Path(self.path, "meta.yaml").open('w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def _get_classname(self, obj):
        cls = type(obj)
        module = cls.__module__
        name = cls.__qualname__
        if module is not None and module != "__builtin__":
            name = module + "." + name
        return name
