# Copyright 2022 EdgeCortix Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Mera Deploy Project utilities."""

import os
import fcntl
import shutil
import logging
import yaml
import json

from pathlib import Path
from enum import Enum


logger = logging.getLogger("mera")
logger.addHandler(logging.NullHandler())

class ArtifactFileType(Enum):
    YAML = "yaml"
    JSON = "json"
    BIN  = "binary"
    TXT  = "text"
    FILE = "file"


class Target(Enum):
    """List of possible Mera Target values."""
    IP = ("IP", False, False)                                #: Target HW accelerator. Valid for `arm` and `x86` architectures.
    Interpreter = ("Interpreter", True, True)                #: Target sw interpretation of the model in floating point. Only valid for `x86`
    InterpreterHw = ("InterpreterHw", True, False)           #: Target sw interpretation of the model. Only valid for `x86`
    Simulator = ("Simulator", True, False)                   #: Target sw simulation of the IP model. Only valid for `x86`
    VerilatorSimulator = ("VerilatorSimulator", True, False) #: Target hw emulation of the IP model. Only valid for `x86`
    InterpreterBf16 = ("InterpreterBf16", True, True)        #: Target sw interpretation of the model in BF16. Only valid for `x86`
    InterpreterHwBf16 = ("InterpreterHwBf16", True, True)    #: Target IP sw interpretation of the model in BF16. Only valid for `x86`
    SimulatorBf16 = ("SimulatorBf16", True, True)            #: Target sw simulation of the IP BF16 model. Only valid for `x86`

    def __init__(self, str_val, x86_only, uses_fp32_flow):
        self.str_val = str_val
        self.x86_only = x86_only
        self.uses_fp32_flow = uses_fp32_flow

    def __str__(self):
        return self.str_val


class Layout(Enum):
    """List of possible data layouts"""
    NCHW = 'NCHW' #: N batches, Channels, Height, Width.
    NHWC = 'NHWC' #: N batches, Height, Width, Channels.


class MeraDeployProject(object):
    """Class for handling a Mera deployment project."""
    _MDP_INFO_PATH = 'project.mdp'
    _MDP_FILE_VERS = 1

    def __init__(self, root_path):
        self.root_path = root_path
        self.log_dir = root_path / 'logs'
        self.log_dir.mkdir(exist_ok=True)
        self.cwd_stack = [self.root_path]
        self.mdp_prj_file = root_path / MeraDeployProject._MDP_INFO_PATH
        self.__load_mdp_file(self.root_path, self.mdp_prj_file)
        self.is_locked = False


    def _lock(self):
        self.lockfd = os.open(self.root_path, os.O_RDONLY)
        try:
            fcntl.flock(self.lockfd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.is_locked = True
        except BlockingIOError:
            raise ValueError(f"Mera project on path '{self.root_path}' is currently being used by another process.")

    def _unlock(self):
        os.close(self.lockfd)
        self.is_locked = False
        self.__save_mdp_file()

    def __save_mdp_file(self):
        mdp_contents = {}
        mdp_contents["prj_version"] = MeraDeployProject._MDP_FILE_VERS
        mdp_contents["artifact_list"] = {k : [str(x.relative_to(self.root_path)) for x in v] for k, v in self.prj_artifacts.items()}
        with open(self.mdp_prj_file, 'w') as w:
            w.write(yaml.safe_dump(mdp_contents))

    def __load_mdp_file(self, root, prj_file):
        self.prj_artifacts = {}
        if prj_file.exists():
            data = yaml.safe_load(prj_file.read_text())
            for sec, files in data["artifact_list"].items():
                self.prj_artifacts[sec] = {Path(root / p) for p in files if Path(root / p ).exists()}
        return set()

    def save_artifact(self, filename, file_type : ArtifactFileType, section, data = None, abs_path = None):
        dst_dir = Path((self.root_path / abs_path) if abs_path else self.get_cwd())
        dst_dir.mkdir(exist_ok=True)
        _f = Path(filename)

        if file_type is ArtifactFileType.FILE:
            if not _f.exists():
                raise ValueError(f'Could not migrate file artifact from {_f}: File not found.')
            dst_path = dst_dir / _f.name
            shutil.copy(_f, dst_dir)
        else:
            dst_path = dst_dir / _f
            if file_type is ArtifactFileType.YAML:
                _d = yaml.safe_dump(data)
            elif file_type is ArtifactFileType.JSON:
                _d = json.dumps(data, indent=4)
            else:
                _d = data
            with open(dst_path, 'w' if file_type is not ArtifactFileType.BIN else 'wb') as w:
                w.write(_d)
        logger.debug(f"Saved {file_type.value} artifact '{_f}' into {dst_dir}")
        self.add_artifact([(section, dst_path)])
        return dst_path

    def add_artifact(self, artifacts):
        for x in artifacts:
            sec, path = x
            if not path.exists():
                raise ValueError(f'Tried to register an artifact that does not exist: {path}')
            if sec not in self.prj_artifacts:
                self.prj_artifacts[sec] = set()
            self.prj_artifacts[sec].add(Path(path))

    def get_artifact(self, section, name):
        art_list = self.prj_artifacts.get(section, set())
        search = [x for x in art_list if Path(x).name == name]
        if not search:
            raise FileNotFoundError(f'Could not find artifact {section}:{name} in the project')
        if len(search) != 1:
            raise ValueError(f'Multiple artifacts found for {section}:{name} -> {search}')
        return search[0]

    def has_artifact_section(self, section):
        return section in self.prj_artifacts

    def pushd(self, path, abs = False):
        _p = Path(self.root_path if abs else self.get_cwd()) / path
        logger.debug(f'(pushd) Running in {path}')
        self.cwd_stack.append(_p)
        _p.mkdir(exist_ok=True)

    def popd(self):
        if len(self.cwd_stack) > 1:
            self.cwd_stack.pop()
        logger.debug(f'(popd) cwd is now {self.get_cwd()}')
        return self.get_cwd()

    def get_cwd(self):
        return self.cwd_stack[-1]

    def get_log_dir(self):
        return str(self.log_dir)


def is_mera_project(path : str) -> bool:
    """Returns whether a provided path is a MeraProject or not

    :param path: Path to check for project existence
    :return: Whether the path belongs to a project
    """
    p = Path(path)
    return p.is_dir() and (p / MeraDeployProject._MDP_INFO_PATH).exists()


def _create_mera_project(path : str, overwrite : bool = False) -> MeraDeployProject:
    """Creates a Mera Deploy Project (mdp) in the designated directory

    :param path: Path to the location where the Mera Deploy Project should be stored
    :param overwrite: Whether to remove the contents of the directory if it already exists
    :return: The project handler object.
    """
    root_path = Path(path).resolve()

    if root_path.exists() and not root_path.is_dir():
        raise ValueError(f'Mera project path {root_path} is not a directory')

    prj_file = root_path / MeraDeployProject._MDP_INFO_PATH
    if prj_file.exists() and overwrite:
        shutil.rmtree(root_path)
    root_path.mkdir(parents=True, exist_ok=True)
    return MeraDeployProject(root_path)
