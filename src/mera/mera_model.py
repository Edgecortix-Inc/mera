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

from typing import Tuple, List
import numpy as np
import json

from pathlib import Path
from enum import Enum
from .deploy_project import ArtifactFileType, logger, Layout
from .model import InputDescription, InputDescriptionContainer

class MeraModel:
    """Base class representing a ML model compatible with MERA deployment project."""

    def __init__(self, prj, model_name, model_path, input_desc, use_prequantize_input):
        self.model_name = model_name
        self.model_path_orig = model_path
        self.input_desc = InputDescriptionContainer(input_desc)
        self.use_prequantize_input = use_prequantize_input

        self.prj = prj
        if prj:
            prj.pushd('model', abs=True)
            if model_path:
                self.model_path = prj.save_artifact(model_path, ArtifactFileType.FILE, 'model')
            prj.save_artifact('input_desc.json', ArtifactFileType.JSON, 'model', self.input_desc.to_dict())
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
            if self.input_desc.num_inputs > 1:
                raise ValueError(f'Multiple input variables exist in {self.model_name}. '
                    + 'Please specify using "input_name" argument.')
            return list(self.input_desc.all_inputs.values())[0].input_shape
        else:
            return self.input_desc.input_desc.get(input_name).input_shape


class MeraModelPytorch(MeraModel):
    """Specialization of MeraModel for a PyTorch ML model."""

    def __init__(self, prj, model_name, model_path, input_desc, layout, use_prequantize_input):
        super().__init__(prj, model_name, model_path, input_desc, use_prequantize_input)
        self.layout = layout

    def _load_model_tvm(self):
        logger.info(f"Loading PyTorch model {self.model_name} for TVM")

        from torch.jit import load as __torch_load
        from torch import device as __device
        from tvm.relay.frontend import from_pytorch as __tvm_from_pytorch

        model_raw = __torch_load(str(self.model_path), map_location=__device('cpu'))
        return __tvm_from_pytorch(model_raw, self.input_desc._for_tvm_pytorch(), layout=str(self.layout.value))


class MeraModelOnnx(MeraModel):
    """Specialization of MeraModel for a ONNX ML model."""

    def __init__(self, prj, model_name, model_path, batch_num):
        super().__init__(prj, model_name, model_path,
            MeraModelOnnx.__get_input_desc(model_path, batch_num), False)
        self.batch_num = batch_num

    def _resolve_model(model_path, batch_num):
        import onnx
        model = onnx.load(str(model_path))
        # Resolve symbolic batch num
        for _input in model.graph.input:
            dim_obj = _input.type.tensor_type.shape.dim
            if dim_obj:
                n_dim = dim_obj[0]
                if n_dim.dim_param != '' and not n_dim.dim_param.isnumeric():
                    # Change symbolic batch with 1
                    n_dim.dim_value = int(batch_num)
        return model

    def __get_input_desc(model_path, batch_num):
        model = MeraModelOnnx._resolve_model(model_path, batch_num)
        from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
        i_desc = []
        for _input in model.graph.input:
            dim_obj = _input.type.tensor_type.shape.dim
            i_name = str(_input.name)
            i_dim = tuple([int(x.dim_value) for x in dim_obj])
            i_type = TENSOR_TYPE_TO_NP_TYPE[_input.type.tensor_type.elem_type]
            i_desc.append(InputDescription(i_name, i_dim, i_type))
        return i_desc

    def _load_model_tvm(self):
        logger.info(f"Loading ONNX model {self.model_name} for TVM")
        model = MeraModelOnnx._resolve_model(self.model_path, self.batch_num)
        from tvm.relay.frontend import from_onnx as __tvm_from_onnx
        return __tvm_from_onnx(model)

class MeraModelTflite(MeraModel):
    """Specialization of MeraModel for a TFLite ML model."""

    def __init__(self, prj, model_name, model_path, use_prequantize_input):
        super().__init__(prj, model_name, model_path, MeraModelTflite.__get_input_desc(model_path), use_prequantize_input)

    def __get_input_desc(model_path):
        import tensorflow.lite
        interpreter = tensorflow.lite.Interpreter(model_path=str(model_path))

        input_desc = []
        for in_details in interpreter.get_input_details():
            input_desc.append(
                InputDescription(in_details['name'], tuple([int(x) for x in in_details['shape']]), np.dtype(in_details['dtype']))
            )
        return input_desc

    def _load_model_tvm(self):
        logger.info(f"Loading TFLite model {self.model_name} for TVM")

        from tflite import Model as __tfliteModel
        from tvm.relay.frontend import from_tflite as __tvm_from_tflite

        model_raw = self.model_path.read_bytes()
        model = __tfliteModel.GetRootAsModel(model_raw, 0)
        shape_dict, dtype_dict = self.input_desc._for_tvm_tflite()

        return __tvm_from_tflite(model, shape_dict=shape_dict, dtype_dict=dtype_dict)

    def _get_mera_aux_config(self):
        return {"weight_layout" : "HWIO", "prequantize_input" : self.use_prequantize_input}


class MeraModelQuantized(MeraModel):
    """MeraModel class of a model quantized with MERA tools."""

    def __init__(self, prj, model_name, model_path):
        from .mera_quantizer import QuantizedMeraModelResult
        self.model_data = QuantizedMeraModelResult.load(model_path)
        super().__init__(prj, model_name, model_path, self.model_data.input_desc, False)
        if prj:
            prj.pushd("model", abs=True)
            prj.save_artifact("quantize_info.json", ArtifactFileType.JSON, 'model', json.dumps(self.model_data.q_params))
            prj.popd()

    def _load_model_tvm(self):
        return self.model_data.qtzed_mod, self.model_data.params


class MeraModelFused(MeraModel):
    """Specialization of MeraModel for a fused model."""

    class FusedModelFrontend(Enum):
        PYTORCH = 'pytorch'
        TFLITE = 'tflite'
        QTZ_MERA = 'qtz_mera'

    def __validate_fused_model(mera_models, share_input):
        assert len(mera_models) > 1, "The input model list should have at least 2 models."
        def __check_is_same(models, model_type):
            if isinstance(models[0], model_type):
                for mera_model in models[1:]:
                    assert isinstance(mera_model, model_type),  \
                        "All models should be from the same frontend (either Pytorch, TFLite or quantized MERA)."
                return True
            return False
        if not (__check_is_same(mera_models, MeraModelPytorch) or \
                __check_is_same(mera_models, MeraModelTflite) or \
                __check_is_same(mera_models, MeraModelQuantized)):
            raise ValueError("Could not find valid models for fusion. "
                + "Please first convert the models into MERA compatible format using mera.ModelLoader")
        if share_input:
            for m in mera_models:
                assert m.input_desc.num_inputs == 1, "When share_input is True, each model should have exactly one input."
            for mera_model in mera_models[1:]:
                assert mera_model.layout == mera_models[0].layout, \
                    "When share_input is True, all models should have the same layout."
        else:
            for mera_model in mera_models[1:]:
                assert mera_model.use_prequantize_input == mera_models[0].use_prequantize_input, \
                    "When share_input is False, all models should have the same value of use_prequantize_input."

    def __get_use_prequantize_input_val(mera_models, share_input):
        if share_input:
            for mera_model in mera_models:
                if mera_model.use_prequantize_input:
                    logger.info("When share_input is True, use_prequantize_input is set to True if one of "
                                 "the models has use_prequantize_input set to True.")
                    return True
            return False
        return mera_models[0].use_prequantize_input

    def __init__(self, prj, mera_models : List[MeraModel], share_input : bool = False):
        MeraModelFused.__validate_fused_model(mera_models, share_input)
        MeraModel.__init__(self, prj, "fused" + "_".join([str(m.model_name) for m in mera_models]),
            None, InputDescriptionContainer._join([m.input_desc for m in mera_models]),
            MeraModelFused.__get_use_prequantize_input_val(mera_models, share_input))

        __frontend_map = {
            MeraModelPytorch : MeraModelFused.FusedModelFrontend.PYTORCH,
            MeraModelTflite: MeraModelFused.FusedModelFrontend.TFLITE,
            MeraModelQuantized: MeraModelFused.FusedModelFrontend.QTZ_MERA
        }
        self.frontend = __frontend_map.get(type(mera_models[0]))
        self.mera_models = mera_models
        self.share_input = share_input

    def _load_model_tvm(self):
        mod_list = []
        params_list = []
        for mera_model in self.mera_models:
            mod, params = mera_model._load_model_tvm()
            mod_list.append(mod)
            params_list.append(params)

        from meratvm_internal.multi_networks import fuse_multi_networks
        fused_mod, fused_params = fuse_multi_networks(mod_list, params_list, self.share_input)
        return fused_mod, fused_params

    def _get_mera_aux_config(self):
        if self.frontend == MeraModelFused.FusedModelFrontend.TFLITE:
            return {"weight_layout" : "HWIO", "prequantize_input" : self.use_prequantize_input}
        else:
            return {"prequantize_input" : self.use_prequantize_input}


class ModelLoader:
    """Utility class for loading and converting ML models into models compatible with MERA

    :param deployer: Reference to a MERA deployer class, if None is provided,
        information about the model will not be added to the deployment project.
    :type deployer: mera.deploy.TVMDeployer
    """
    def __init__(self, deployer = None):
        self.prj = deployer.prj if deployer else None

    def __check_model(self, model_path, model_name):
        m_loc = Path(model_path).resolve()
        if not model_name:
            model_name = m_loc.stem

        if not m_loc.exists():
            raise ValueError(f'Model file {model_path} does not exist or could not be accessed')
        return model_name

    def from_tflite(self, model_path : str, model_name : str = None, use_prequantize_input : bool = False) -> MeraModelTflite:
        """Converts a tensorflow model in TFLite format into a compatible model for MERA.

        :param model_path: Path to the tensorflow model file in TFLite format
        :param model_name: Display name of the model being deployed.
            Will default to the stem name of the model file if not provided.
        :param use_prequantize_input: Whether input is provided prequantized, or not. Defaults to False
        :return: The input model compatible with MERA.
        """
        model_name = self.__check_model(model_path, model_name)
        return MeraModelTflite(self.prj, model_name, model_path, use_prequantize_input)

    def from_pytorch(self, model_path : str, input_desc, model_name : str = None, layout : Layout = Layout.NHWC,
            use_prequantize_input : bool = False) -> MeraModelPytorch:
        """Converts a PyTorch model in TorchScript format into a compatible model for MERA.

        :param model_path: Path to the PyTorch model file in TorchScript format
        :param input_desc: Map of input names and their dimensions and types.
            Expects a format of {input_name : (input_size, input_type)}
        :param model_name: Display name of the model being deployed.
            Will default to the stem name of the model file if not provided.
        :param layout: Data layout of the model being loaded. Defaults to NHWC layout
        :param use_prequantize_input: Whether input is provided prequantized, or not. Defaults to False

        :return: The input model compatible with MERA.
        """
        model_name = self.__check_model(model_path, model_name)
        return MeraModelPytorch(self.prj, model_name, model_path, input_desc, layout, use_prequantize_input)

    def from_onnx(self, model_path : str, model_name : str = None, layout : Layout = Layout.NHWC,
        batch_num : int = 1) -> MeraModelOnnx:
        """Converts a ONNX model into a compatible model for MERA.
        NOTE this loader is best optimised for float models using op_set=12

        :param model_path: Path to the ONNX model file.
        :param model_name: Display name of the model being deployed.
            Will default to the stem name of the model file if not provided.
        :param layout: Data layout of the model being loaded. Defaults to NHWC layout
        :param batch_num: If the model contains symbolic batch numbers, loads it resolving its value to
            the parameter provided. Defaults to 1.

        :return: The input model compatible with MERA.
        """
        model_name = self.__check_model(model_path, model_name)
        return MeraModelOnnx(self.prj, model_name, model_path, batch_num)

    def from_quantized_mera(self, model_path : str, model_name : str = None):
        """Converts a previously quantized MERA model into a compatible deployable model.

        :param model_path: Path to the MERA model file
        :param model_name: Display name of the model being deployed.
            Will default to the stem name of the model file if not provided.

        :return: The input model compatible with MERA.
        """
        model_name = self.__check_model(model_path, model_name)
        return MeraModelQuantized(self.prj, model_name, model_path)

    def fuse_models(self, mera_models : Tuple[MeraModel], share_input : bool = False) -> MeraModelFused:
        """Fusing multiple MERA models into a single model for compilation and deployment.
           This is especially useful for fully utilizing the compute resources of a large platform.
           The inputs of the fused model are the concatenation of the inputs of the models to be fused.
           Similarly, the outputs of the fused model are the concatenation of the outputs of the models to be fused.
           For example, let's suppose `mera_models` has two models, m1 and m2, then for the fused model,
           the inputs are [m1 inputs, m2 inputs] and the outputs are [m1 outputs, m2 outputs].
           When each model in `mera_models` has one input and `share_input` is True, the fused model has one input.

        :param mera_models: List of MERA models to be fused.
        :param share_input: Whether the models share input or not.

        :return: The fused model.
        """
        try:
            from meratvm_internal.multi_networks import fuse_multi_networks
        except ModuleNotFoundError:
            raise ValueError("The currently installed MERA package belongs to a version with limited premium features.\n"
                "As such, the multi network fusion feature is not enabled on this version.\n"
                "Please get in touch with the MERA team in order to get the full version.")
        return MeraModelFused(self.prj, mera_models, share_input)
