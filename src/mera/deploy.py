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
"""Mera Deployer classes"""

import os
import time
import platform
import numpy as np

from datetime import date

from .deploy_project import ArtifactFileType, Target, logger, _create_mera_project
from .mera_model import MeraModel
from .version import __version__
from .mera_deployment import MeraTvmDeployment, MeraTvmPrjDeployment
from .mera_platform import Platform


class _DeployerBase:
    """Base class for Mera deployer handler"""

    def __init__(self, output_dir : str, overwrite : bool = False):
        """Create a new deployment project with a given toolchain

        :param output_dir: Output directory relative to cwd where the project should be deployed
        :param overwrite: Whether the folder should be wiped before starting a new deployment. Defaults to false
        """
        self.prj = _create_mera_project(output_dir, overwrite)

    def __enter__(self):
        self.prj._lock()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.prj._unlock()


class TVMDeployer(_DeployerBase):
    """Using MERA deployer targetting the TVM compiler stack:"""

    def _save_compile_metrics(self, compile_time):
        metrics = {}
        metrics['compile_time'] = compile_time
        metrics['compile_date'] = date.today().strftime("%d/%m/%Y")
        # TODO - Add more metrics
        self.prj.save_artifact('compile_metrics.json', ArtifactFileType.JSON, 'metrics', metrics)

    def deploy(self, model : MeraModel, mera_platform : Platform = Platform.DNAF200L0003, build_config = {},
        target : Target = Target.Simulator, host_arch : str = None) -> MeraTvmDeployment:
        """Launches the compilation of a MERA project for a MERA model using the TVM stack.

        :param model: Model object loaded from mera.ModelLoader
        :param mera_platform: MERA platform architecture enum value
        :param build_config: MERA build configuration dict
        :param target: MERA build target
        :param host_arch: Host arch to deploy for. If unset, it will pick the current host platform, 
            provide a value to override the setting
        :return: The object representing the result of a MERA deployment
        """
        if not isinstance(model, MeraModel):
            raise ValueError(f'Model is not of MeraModel type.')

        if isinstance(target, str):
            target = Target[target]
        elif not isinstance(target, Target):
            raise ValueError(f'target parameter {target} could not be interpreted as a valid Target')
        target_str = target.str_val
        x86_only = target.x86_only

        if isinstance(mera_platform, Platform):
            arch_val = mera_platform.value
        else:
            arch_val = mera_platform

        __PRCS_MAP = {
            'x86_64' : 'x86',
            'i386' : 'x86',
            'AMD64' : 'x86'
            # TODO - Missing mapping for ARM
        }
        __SUPPORTED_ARCH = ['x86', 'arm']
        if not host_arch:
            _prcs = platform.processor()
            host_arch = __PRCS_MAP.get(_prcs, _prcs)

        if host_arch not in __SUPPORTED_ARCH:
            raise ValueError(f'Unsupported host architecture "{host_arch}". '
            f'Only [{" ".join(__SUPPORTED_ARCH)}] architectures are supported.')

        if host_arch != 'x86' and x86_only:
            raise ValueError(f'Selected host_arch="{host_arch}", but target "{target_str}" only supports x86 architectures')

        from tvm.relay import mera as _mera
        logger.info(f" *** mera v{__version__} ***")
        logger.info(f"Starting deployment of model '{model.model_name}'...")

        # Change directory to target's build subdir
        self.prj.pushd("build", abs=True)
        self.prj.pushd(target_str)

        # Setup mera logging
        os.environ["GLOG_log_dir"] = self.prj.get_log_dir()

        # Save compile input artifacts
        self.prj.save_artifact('build_config.yaml', ArtifactFileType.YAML, target_str, build_config)
        mera_compiler_cfg = {**build_config, "arch" : arch_val}
        self.prj.save_artifact('mera_cfg.json', ArtifactFileType.JSON, target_str, mera_compiler_cfg)
        mod, params = model._load_model_tvm()
        mera_compiler_cfg['target'] = target_str
        with _mera.build_config(**mera_compiler_cfg):
            logger.info(f'Compiling Mera model...')
            self.prj.pushd('result')
            tm_start = time.time()
            if target.uses_fp32_flow:
                _mera.build_fp32(mod, params, target_str, host_arch=host_arch, output_dir=self.prj.get_cwd())
            else:
                _mera.build(mod, params, output_dir=self.prj.get_cwd(),
                    host_arch=host_arch, layout='NHWC', aux_config=model._get_mera_aux_config())
            tm_end = time.time()
            to_target_artifact = lambda a : (target_str, self.prj.get_cwd() / a)
            self.prj.add_artifact([to_target_artifact(x) for x in ['deploy.so', 'deploy.json', 'deploy.params']])

            time_taken = tm_end - tm_start
            logger.info(f'Compilation finished successfully. Took {time.strftime("%Hh%Mm%Ss", time.gmtime(time_taken))}')
            self._save_compile_metrics(time_taken)
            self.prj.popd() # result

        self.prj.popd() # target
        self.prj.popd() # build

        logger.info(f'Deployment completed')
        lib_path = self.prj.get_artifact(target_str, 'deploy.so')
        params_path = self.prj.get_artifact(target_str, 'deploy.params')
        lib_json_path = self.prj.get_artifact(target_str, 'deploy.json')
        return MeraTvmPrjDeployment(lib_path, params_path, lib_json_path, self.prj)

    def save_data_file(self, filename : str, data : np.ndarray):
        """
        Helper function to store a given array into the project to later be used by other applications.

        :param filename: Filename to give to the saved array
        :param data: The data array to save
        """
        self.prj.save_artifact(filename, ArtifactFileType.BIN, 'data', data, 'data')

    def load_data_file(self, filename : str) -> np.ndarray:
        """
        Helper function to load a given array stored in the project with :func:`save_data_file()`

        :param filename: File name of the saved array
        :return: The data array loaded from teh project, or an exception if it could not be found.
        """
        return np.load(self.prj.get_artifact('data', filename))
