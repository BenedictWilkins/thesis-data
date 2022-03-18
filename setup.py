#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 16-03-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

from setuptools import setup, find_packages

setup(name='thesisdata',
        version='0.0.1',
        description='',
        url='',
        author='Benedict Wilkins',
        author_email='benrjw@gmail.com',
        packages=find_packages(),
        install_requires=[
            'numpy', 'hydra-core', 'kaggle', 'torch', 'torchvision',
            # 'jnu @ ...'
            #'gymu @ git+https://git@github.com/BenedictWilkins/gymu.git',
        ],
        entry_points={
            "gym.envs": [
                "thesis = thesisdata.gym:register_entry_point"
            ]
        },
        zip_safe=False)