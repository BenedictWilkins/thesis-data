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

class FileResolver:

    def __init__(self, dirs, suffix=['.tar*']):
        self.base = [pathlib.Path(d).expanduser().resolve() for d in dirs]
        def _proper_suffix(s):
            if not s.startswith("."):
                return "." + s
            return s
        self.suffix = [_proper_suffix(s) for s in suffix]
        # validate paths, dir is ok
        for path in self.base:
            if not path.exists():
                raise ValueError(f"Path {path} does not exist.")
            if path.is_file() and not any([re.match(s, path.suffix) is not None for s in suffix]):
                raise ValueError(f"Path {path} does not have the required suffix {self.suffix}.")
        
    @property
    def files(self):
        files = []
        for path in self.base:
            if path.is_file():
                files.append(path)
            else:
                for suffix in self.suffix:
                    files.extend(glob.glob(str(pathlib.PurePath(path, f"*{suffix}")), recursive=True))
        return files

class SymlinkFileResolver(FileResolver):

    def __init__(self, base, alias, **kwargs):
        base = pathlib.Path(base).expanduser().resolve()
        if not base.exists():
            raise ValueError(f"Path {base} does not exists, could not create symlink.")

        alias = pathlib.Path(alias).expanduser().resolve()
        if alias.exists():
            raise ValueError(f"Path {alias} already exists, could not create symlink.")
        
        os.symlink(base, alias)
        super().__init__([base], **kwargs)


'''
def resolve_files(self, path, data_split):
      """ 
         Search for train, validate and test directories, otherwise split the data according to cfg.dataset.split.
         Args:
            path (pathlib.Path, str): path to search.
      """
      if path is None:
         return [], [], []
      path = pathlib.Path(path)
      if not path.exists():
         raise ValueError(f"Dataset path: '{path}' doesnt exist.")
      def _find(dirs, *labels):
         found = next((v for k,v in dirs.items() if k in labels), None)
         if found is not None:
            Logger.info(f"Found data directory: {found}")
            found = pathlib.PurePath(found, "**/*.tar*")
            return list(sorted(glob.glob(str(found), recursive=True)))
         return []
      # search for train/validate/test directories for data split.
      dirs = {f.name:f for f in path.iterdir() if f.is_dir()}
      train_files = _find(dirs, 'train', 'training')
      test_files = _find(dirs, 'test', 'testing')
      validate_files = _find(dirs, 'val', 'validate', 'validation')
      files = train_files + test_files + validate_files
      if len(files) == 0: # no directories/files were found... use files from the current folder and manually split the data.
         path = pathlib.PurePath(path, "**/*.tar*")
         files = list(sorted(glob.glob(str(path), recursive=True)))
         assert len(files) > 0 # didn't find any files...
         Logger.info(f"Found data directory: {path}")
         # reversing the split ensures there will always be 1 training file if there is a fractional train/val/test split.
         split = np.cumsum(np.array(list(reversed(data_split))) * len(files)).astype(np.int64)[:-1]
         test_files, validate_files, train_files = [x.tolist() for x in np.split(np.array(files), split)]

      Logger.debug("Train files:\n" + "   \n".join(train_files))
      Logger.debug("Validate files:\n" + "    \n".join(validate_files))
      Logger.debug("Test files:\n" + "    \n".join(test_files))

      Logger.info(f"Found: {len(files)} files, {len(train_files)} train files, {len(validate_files)} validate files, {len(test_files)} test files.")
      assert len(files) > 0
      assert len(train_files) > 0
      return train_files, validate_files, test_files
'''