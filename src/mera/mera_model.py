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
"""Mera Model classes."""

from typing import Tuple
import numpy as np

from pathlib import Path
from .deploy_project import ArtifactFileType, logger


class MeraModel:
    """Base class representing a ML model compatible with Mera stack."""

    def __init__(self, prj, model_name, model_path, input_desc, use_prequantize_input):
        self.prj = prj
        self.model_name = model_name
        self.model_path_orig = model_path
        self.input_desc = input_desc
        self.use_prequantize_input = use_prequantize_input

        prj.pushd('model', abs=True)
        self.model_path = prj.save_artifact(model_path, ArtifactFileType.FILE, 'model')
        prj.save_artifact('input_desc.json', ArtifactFileType.JSON, 'model', self.input_desc)
        prj.popd()

    def _load_model_tvm(self):
        # To be overriden by child classes
        raise NotImplementedError()

    def _get_mera_aux_config(self):
        return {"prequantize_input" : self.use_prequantize_input}

    def get_input_shape(self, input_name : str = None) -> Tuple[int]:
        """Utility class to query the shape of an input variable of the model

        :param input_name: Specifies which input to get the shape from. If unset, assumes there is only one input.
        :return: A tuple with 4 items representing the shape of the input variable in the model.
        """
        if not input_name:
            if len(self.input_desc) > 1:
                raise ValueError(f'Multiple input variables exist in {self.model_name}. '
                    + 'Please specify using "input_name" argument.')
            return list(self.input_desc.values())[0][0]
        else:
            return self.input_desc[input_name][0]



class MeraModelPytorch(MeraModel):
    """Specialization of MeraModel for a PyTorch ML model."""

    def _load_model_tvm(self):
        logger.info(f"Loading PyTorch model {self.model_name} for TVM")

        from torch.jit import load as __torch_load
        from tvm.relay.frontend import from_pytorch as __tvm_from_pytorch

        model_raw = __torch_load(str(self.model_path))
        input_dims = [(k, v) for k, v in self.input_desc.items()]
        return __tvm_from_pytorch(model_raw, input_dims, layout="NHWC")


class MeraModelTflite(MeraModel):
    """Specialization of MeraModel for a TFLite ML model."""

    def __init__(self, prj, model_name, model_path, use_prequantize_input):
        super().__init__(prj, model_name, model_path, MeraModelTflite.__get_input_desc(model_path), use_prequantize_input)

    def __get_input_desc(model_path):
        import tensorflow.lite
        interpreter = tensorflow.lite.Interpreter(model_path=str(model_path))

        input_desc = {}
        for in_details in interpreter.get_input_details():
            input_name = in_details['name']
            input_shape = tuple([int(x) for x in in_details['shape']])
            input_type = np.dtype(in_details['dtype']).name
            input_desc[input_name] = (input_shape, input_type)
        return input_desc

    def _load_model_tvm(self):
        logger.info(f"Loading TFLite model {self.model_name} for TVM")

        from tflite import Model as __tfliteModel
        from tvm.relay.frontend import from_tflite as __tvm_from_tflite

        model_raw = self.model_path.read_bytes()
        model = __tfliteModel.GetRootAsModel(model_raw, 0)
        shape_dict = {k : v[0] for k,v in self.input_desc.items()}
        dtype_dict = {k : np.dtype(v[1]).name for k,v in self.input_desc.items()}

        return __tvm_from_tflite(model, shape_dict=shape_dict, dtype_dict=dtype_dict)

    def _get_mera_aux_config(self):
        return {"weight_layout" : "HWIO", "prequantize_input" : self.use_prequantize_input}


class ModelLoader:
    """Utility class for loading and converting ML models into models compatible with Mera

    :param deployer: Reference to a Mera deployer class
    :type deployer: mera.deploy.TVMDeployer
    """
    def __init__(self, deployer):
        self.prj = deployer.prj

    def __check_model(self, model_path, model_name):
        m_loc = Path(model_path).resolve()
        if not model_name:
            model_name = m_loc.stem
        
        if not m_loc.exists():
            raise ValueError(f'Model file {model_path} does not exist or could not be accessed')
        return model_name

    def from_tflite(self, model_path : str, model_name : str = None, use_prequantize_input : bool = False) -> MeraModelTflite:
        """Converts a tensorflow model in TFLite format into a compatible model for Mera.

        :param model_path: Path to the tensorflow model file in TFLite format
        :param model_name: Display name of the model being deployed.
            Will default to the stem name of the model file if not provided.
        :param use_prequantize_input: Whether input is provided prequantized, or not. Defaults to False
        :return: The input model compatible with Mera.
        """
        model_name = self.__check_model(model_path, model_name)
        return MeraModelTflite(self.prj, model_name, model_path, use_prequantize_input)

    def from_pytorch(self, model_path : str, input_desc, model_name : str = None, use_prequantize_input : bool = False) -> MeraModelPytorch:
        """Converts a PyTorch model in TorchScript format into a compatible model for Mera.

        :param model_path: Path to the PyTorch model file in TorchScript format
        :param input_desc: Map of input names and their dimensions and types.
            Expects a format of {input_name : (input_size, input_type)}
        :param model_name: Display name of the model being deployed.
            Will default to the stem name of the model file if not provided.
        :param use_prequantize_input: Whether input is provided prequantized, or not. Defaults to False

        :return:  The input model compatible with Mera.
        """
        model_name = self.__check_model(model_path, model_name)
        return MeraModelPytorch(self.prj, model_name, model_path, input_desc, use_prequantize_input)
