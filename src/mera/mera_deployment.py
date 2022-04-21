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
"""Mera Deployment classes"""

import numpy as np

from typing import Dict, List
from pathlib import Path
from .deploy_project import is_mera_project, _create_mera_project, Target, logger


class MeraTvmModelRunner:
    def __init__(self, rt_mod):
        self.rt_mod = rt_mod

    def set_input(self, data : np.ndarray):
        """Sets the input data for running

        :param data: Input numpy data tensor or dict of input numpy data tensors if the model has more than one input.
            Setting multiple inputs should have the format `{input_name : input_data}`
        """
        if not isinstance(data, dict):
            d_ = [data]
        else:
            d_ = [None] * self.rt_mod.get_num_inputs()
            for k, d in data.items():
                d_[self.rt_mod.get_input_index(k)] = d
        [self.rt_mod.set_input(i, d) for i, d in enumerate(d_)]
        logger.info(f'Set {len(d_)} input data variables')
        return self

    def run(self) -> None:
        """Runs the model with the specified input data. :func:`set_input()` needs to be called before :func:`run()`"""
        self.rt_mod.run()
        return self

    def get_output(self, output_idx : int = 0) -> np.ndarray:
        """Returns the output tensor given an output id index. :func:`run()` needs to be called before :func:`get_output()`

        :param output_idx: Index of output variable to query
        :return: Output tensor values in numpy format
        """
        return self.rt_mod.get_output(output_idx).asnumpy()

    def get_outputs(self) -> List[np.ndarray]:
        """Returns a list of all output tensors. Equivalent to :func:`get_output()` from `[0, get_num_outputs()]`

        :return: List of output tensor values in numpy format
        """
        return [self.get_output(i) for i in range(self.get_num_outputs())]

    def get_num_outputs(self) -> int:
        """Gets the number of available outputs

        :return: Number of output variables
        """
        return self.rt_mod.get_num_outputs()

    def get_runtime_metrics(self) -> dict:
        """Gets the runtime metrics reported from Mera after a :func:`run()`

        :return: Dictionary of measured metrics
        """
        return self.rt_mod.get_runtime_metrics()


class MeraTvmDeployment:
    def __init__(self, lib_path, params_path, lib_json_path):
        self.lib_path = lib_path
        self.params_path = params_path
        self.lib_json_path = lib_json_path

    def get_runner(self) -> MeraTvmModelRunner:
        """Prepares the model for running with a given target

        :return: Runner object
        """
        from tvm.runtime import load_module as __load_module, cpu as __cpu
        from tvm.contrib.graph_executor import create as __create
        rt_mod = __create(self.lib_json_path.read_text(), __load_module(self.lib_path), __cpu())
        rt_mod.load_params(self.params_path.read_bytes())
        logger.info(f'Created TVM model runner')
        return MeraTvmModelRunner(rt_mod)


class MeraTvmPrjDeployment(MeraTvmDeployment):
    def __init__(self, lib_path, params_path, lib_json_path, prj):
        super().__init__(lib_path, params_path, lib_json_path)
        self.prj = prj


def load_mera_deployment(path : str, target : Target = None) -> MeraTvmDeployment:
    """Loads an already built deployment from a directory

    :param path: Directory of a Mera deployment project or full directory of built mera results
    :param target: If there are multiple targets built in the mera project selects which one.
        Optional if not loading a project or if there is a single target built.
    :return: Reference to deployment object
    """
    p = Path(path).resolve()
    if is_mera_project(p):
        logger.info(f"Loading deployment from Mera project '{p}' ...")
        prj = _create_mera_project(p)
        avail_targets = [x for x in Target if prj.has_artifact_section(x.str_val)]
        if not avail_targets:
            raise ValueError(f'Could not find a valid deployment avaialble in project {p}')
        if len(avail_targets) > 1 and target not in avail_targets:
            raise ValueError(f'Could not find target {target} from built targets [{" ".join([x.str_val for x in avail_targets])}]')
        t = avail_targets[0] if len(avail_targets) == 1 else target

        lib_path = prj.get_artifact(t.str_val, 'deploy.so')
        params_path = prj.get_artifact(t.str_val, 'deploy.params')
        lib_json_path = prj.get_artifact(t.str_val, 'deploy.json')
        logger.info(f'Successfully loaded deployment project')
        return MeraTvmPrjDeployment(lib_path, params_path, lib_json_path, prj)
    else:
        logger.info(f'Loading deployment from folder {p} ...')
        def __check_file(file):
            f = Path(file)
            if not f.exists():
                raise ValueError(f"Could not find file '{f.name}' in directory {f.parent}")
            return f
        lib_path, params_path, lib_json_path = [__check_file(p / x) for x in ['deploy.so', 'deploy.params', 'deploy.json']]
        logger.info(f'Successfully loaded deployment package')
        return MeraTvmDeployment(lib_path, params_path, lib_json_path)
