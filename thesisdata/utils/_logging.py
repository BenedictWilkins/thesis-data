#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" 
   Created on 16-03-2022
"""
__author__ = "Benedict Wilkins"
__email__ = "benrjw@gmail.com"
__status__ = "Development"

import logging
import sys

Logger = None
def getLogger():
   global Logger
   if Logger is None:
      Logger = logging.getLogger("thesisdata")
      Logger.setLevel(logging.DEBUG)
      if len(Logger.handlers) == 0:
         streamHandler = logging.StreamHandler(sys.stdout)
         formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
         streamHandler.setFormatter(formatter)
         Logger.addHandler(streamHandler)
   return Logger

__all__ = ("getLogger","Logger")