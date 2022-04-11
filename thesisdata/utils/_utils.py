#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 21-03-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import pathlib
from typing import Union, List, Dict

def resolve_class(cls):
    from importlib import import_module
    try:
        module_path, class_name = cls.rsplit('.', 1)
        module = import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(cls)

def get_project_root_directory(root='thesisdata'):
    current_dir = pathlib.Path(__file__)
    return [p for p in current_dir.parents if p.parts[-1]==root][0].parent
