#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 06-04-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import pathlib
import glob
import os
import re
import gym

from typing import Union, List, Dict
from omegaconf import OmegaConf, ListConfig, DictConfig



from ._logging import Logger

def omegaconf_resolver(name, **kwargs): # decorator for registering omegaconf_resolvers
    def _omegaconf_resolver(fun):
        OmegaConf.register_new_resolver(name, fun, **kwargs)
        return fun
    return _omegaconf_resolver

def get_environment_config(env : gym.Env): # TODO this might be moved over to a yaml resolver/constructor... its a little bit tricky to deal with env wrappers...
    spec = {k:v for k,v in env.spec.__dict__.items()}
    spec['env_id'] = env.spec.id
    wrappers = [w for w in get_wrappers(env)]
    config = dict(spec=spec, 
                wrappers=wrappers, 
                action_space=env.action_space, 
                observation_space=env.observation_space)
    if hasattr(env, "get_action_meanings"):
        config['action_meanings'] = env.get_action_meanings()
    return config

def get_wrappers(env):
    import gymu
    import stable_baselines3
    def _get_classname(cls):
        module = cls.__module__
        name = cls.__qualname__
        if module is not None and module != "__builtin__":
            name = module + "." + name
        return name
    while hasattr(env, "env"):
        if isinstance(env, gymu.iter._intercept._InterceptWrapper):
            pass 
        elif isinstance(env, stable_baselines3.common.monitor.Monitor): # dont need these wrappers.
            pass 
        elif isinstance(env, stable_baselines3.common.atari_wrappers.AtariWrapper):
            yield _get_classname(env.__class__)
            while 'atari_wrappers' in _get_classname(env.__class__):
                env = env.env
        else:
            yield _get_classname(env.__class__)
        env = env.env

def download_from_kaggle(path : Union[str, pathlib.Path], urls : Union[List[str],str], force : bool = False):
    """
        Download kaggle dataset(s) to path <PATH>.
        Download path for each url will be <PATH>/<DATASET_NAME> where url = <USER_NAME>/<DATASET_NAME>
        
        Requires kaggle authentication, ensure that ~/.kaggle/kaggle.json contains the relevant information.
    Args:
        path (Union[str, pathlib.Path]): path to download to
        urls (Union[List[str],str]): kaggle urls <USER_NAME>/<DATASET_NAME>
        force (bool, optional): override data if it already exists. Defaults to False.

    Raises:
        ValueError: if urls is empty.
    """
    from kaggle.api.kaggle_api_extended import KaggleApi
    path = pathlib.Path(path)
    if isinstance(urls, str):
        urls = [urls]
    if len(urls) == 0: raise ValueError("Atleast one download URL must be specified.")
    api = KaggleApi()
    api.authenticate() # requires ~/.kaggle/kaggle.json file
    path.mkdir(parents=True, exist_ok=True)
    for url in urls:
        download_path = pathlib.Path(path, url)
        api.dataset_download_files(url, quiet=False, unzip=True, force=force, path=str(download_path))

@omegaconf_resolver('environment', replace=True, use_cache=True)
def configure_environment(cfg):
    """ Get environment configuration for a hydra config file. 
        
        The argument 'cfg' should contain the following: 
        ```
        env_id: <ENV_ID>                # required (str)
        path: <PATH>                    # optional (str)  
        kaggle:                         # optional (dict)
            urls: [<URL>, ...]          # optional (list[str])
            force: <FORCE>              # optional (bool)
            train: <TRAIN>              # optional (list[str])
            validate: <VALIDATE>        # optional (list[str])
            test: <TEST>                # optional (list[str])
            write_meta: <WRITE_META>    # optional (bool)
        ```
        1. If <PATH> is not specified, get the configuration by creating an environment <ENV_ID>. The environment must be registered with gym.
        2. Download the dataset from kaggle if kaggle is specified (see configure_from_kaggle)
        3. Searchs for a 'meta.yaml' file in the given dataset at <PATH>
        4. If 'meta.yaml' cannot be found, it is created using <ENV_ID>
    Args:
        cfg (DictConfig): config data.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    import yaml
    import gym
    from omegaconf import OmegaConf
    # cfg should be the dataset configuration and contain either path or env_id.
    def _strip_yaml_tags(yaml_data):
        result = []
        for line in yaml_data.splitlines():
            idx = line.find("!")
            if idx > -1:
                line = line[:idx]
            result.append(line)
        return '\n'.join(result)
    
    def get_meta_from_environment():
        kwargs = cfg.get("env_kwargs", OmegaConf.create())
        kwargs = OmegaConf.to_container(kwargs)
        env = gym.make(cfg.env_id, **kwargs)
        env.reset() # some environments require this to prevent hanging...
        yaml_data = yaml.dump(get_environment_config(env))
        if hasattr(env, "close"):
            env.close()
        yaml_data = _strip_yaml_tags(yaml_data)
        return OmegaConf.create(yaml_data)

    if not 'path' in cfg:
        return get_meta_from_environment()
    elif 'kaggle' in cfg:
        urls = cfg.kaggle.urls
        if isinstance(urls, str):
            urls = [urls]
        download_from_kaggle(cfg.path, urls, force = cfg.kaggle.get('force', False))
        symlinked(pathlib.Path(cfg.path, 'train'), [pathlib.Path(cfg.path, p) for p in cfg.kaggle.get('train', [])], force=True)
        symlinked(pathlib.Path(cfg.path, 'validate'), [pathlib.Path(cfg.path, p) for p in cfg.kaggle.get('validate', [])], force=True)
        symlinked(pathlib.Path(cfg.path, 'test'), [pathlib.Path(cfg.path, p) for p in cfg.kaggle.get('test', [])], force=True)
    
    metas = [pathlib.Path(p) for p in glob.glob(str(pathlib.PurePath(cfg.path, "**/meta.yaml")), recursive=True)]
    
    if len(metas) == 0: # create meta file for dataset...
        Logger.debug(f"Failed to find meta.yaml, creating it using environment {cfg.env_id}")
        meta_data = get_meta_from_environment()
        meta_file = pathlib.Path(cfg.path, "meta.yaml").open("w")
        yaml.dump(OmegaConf.to_container(meta_data), meta_file)
        meta_file.close()
        return meta_data
    
    Logger.debug(f"Meta file {metas[0]}")
    with metas[0].open('r') as f:
        data =  OmegaConf.create(_strip_yaml_tags(yaml.dump(yaml.safe_load(f))))
        return data

@omegaconf_resolver("symlinked")
def symlinked(base : str, paths : Union[List[str], ListConfig, str], force=True):
    if isinstance(paths, str):
        paths = [paths]
    if len(paths) == 0:
        return ListConfig([])
    base = pathlib.Path(base).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    paths = [pathlib.Path(path).expanduser().resolve() for path in paths]
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Path {path} must exist to create symlink.")
    def shared_relative(paths):
        shortest = min([len(path.parents) for path in paths])
        for i in range(1, shortest):
            if set([str(list(path.parents)[-i]) for path in paths]) == 1:
                return [path.relative_to(paths[0].parents[-i]) for path in paths]
        return [path.relative_to(list(paths[0].parents)[-shortest]) for path in paths]
    relative_paths = shared_relative(paths)
    link_paths = [pathlib.Path(base, path) for path in relative_paths]
    if force: # clear symlinks...
        for link in base.iterdir():
            if link.is_symlink():
                link.unlink()

    for s,d in zip(paths, link_paths):
        d.symlink_to(s)
    return ListConfig([str(link) for link in link_paths])

@omegaconf_resolver('merge')
def omegaconfig_list_merge(x : ListConfig, y : ListConfig):
    """ Merge two lists using omegaconf variable interpolation.

        Example: 
            config.yaml
            ```yaml
            list1 : [0]
            list2 : [1,2,3]
            list3 : ${merge:${list1},${list2}}
            ```
            ```
            >> list1 = [0]
            >> list2 = [1,2,3]
            >> list3 = [0,1,2,3]
            ``
    Args:
        x (ListConfig): list 1
        y (ListConfig): list 2

    Returns:
        ListConfig: merged list
    """
    if not isinstance(x, ListConfig):
        x = ListConfig([x])
    if not isinstance(y, ListConfig):
        y = ListConfig([y])
    return x + y


@omegaconf_resolver('s')
def omegaconfig_slice(x : ListConfig):
    """ 'slice' syntax in omegaconf for creating slice objects.
        Example:
            config.yaml 
            ```yaml
            slice1: ${s:[:1]} 
            ```
            resolves to: 
            ```
            slice1:
                _target_: 'builtins.slice'
                _args_ : [None,1,None]
            ```
        hydra can create the instance using 'instantiate' to produce the slice object: `slice(None,1,None)`
    Args:
        x (_type_): _description_

    Returns:
        _type_: _description_
    """
    assert len(x) == 1 # only single element slices are supported currently TODO
    x = str(x[0]) # its parsed as a list containing a string...
    pattern = re.compile("(-?[0-9]*):(-?[0-9]*):?(-?[0-9]*)|(-?[0-9]*)")
    s = re.findall(pattern, x)[0]
    s = [(int(x) if x != '' else None) for x in s]
    return s[-1] if s[-1] is not None else DictConfig(dict(_target_='builtins.slice', _args_= ListConfig(s[:3]))) # int or slice list


