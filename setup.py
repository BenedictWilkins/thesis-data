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
            #'jnu @ ...'
            #'gymu @ git+https://git@github.com/BenedictWilkins/gymu.git',
            #'stable_baselines3 @ git+https://github.com/DLR-RM/stable-baselines3.git#009bb0549ad0c9c1130309d95529a237e126578c'
        ],
        entry_points={
            "gym.envs": [
                "thesis = thesisdata.environment:register_entry_point"
            ]
        },
        zip_safe=False)