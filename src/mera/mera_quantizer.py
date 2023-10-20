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
"""Mera Quantizer classes"""

import torch
import json
import pickle
import mera
import numpy as np

from argparse import ArgumentError
from typing import Union, List, Tuple
from pathlib import Path
from .deploy_project import logger, Layout, Target
from .quantization_quality import *
from tqdm import tqdm
from enum import Enum
from tempfile import TemporaryDirectory
from torch.nn.modules.module import Module as TorchModule
from torch.jit import ScriptModule as TorchScriptModule

class ModelQuantizerFlow(Enum):
    TORCH = 'torch'
    ONNX = 'onnx'
    TF_LITE = 'tflite'
    MERA_MODEL = 'mera_model'

class QuantizedMeraModelResult:
    """Class that represents the result of a model quantized with the MERA quantizer."""
    __VERSION_PACK = 2

    def __init__(self, input_desc, qtzed_mod, q_params, params, fp32_rtmod):
        self.input_desc = input_desc
        self.qtzed_mod = qtzed_mod
        self.q_params = q_params
        self.params = params
        self.fp32_rtmod = fp32_rtmod

    def save(self, file_name : Union[str, Path]) -> None:
        f_path = Path(file_name).resolve()
        logger.debug(f"Saving quantized MERA model to '{f_path}' ...")
        from tvm.ir import save_json as __tvm_save_json
        ser_tvm_ir = __tvm_save_json(self.qtzed_mod)
        ser_params = {k: v.numpy() for k,v in self.params.items()}
        pickle_data = {
            "version" : QuantizedMeraModelResult.__VERSION_PACK,
            "data" : (self.input_desc, ser_tvm_ir, self.q_params, ser_params)
        }
        with open(f_path, "wb") as f:
            pickle.dump(pickle_data, f)

    def save_qtz_parameters(self, file_name : Union[str, Path]) -> None:
        f_path = Path(file_name).resolve()
        logger.debug(f"Saving MERA quantization parameters to '{f_path}' ...")
        with open(f_path, "w") as f:
            f.write(json.dumps(self.q_params))

    def load(file_name : Union[str, Path]):
        f_path = Path(file_name).resolve()
        if not f_path.is_file():
            raise ValueError(f"Could not open MERA quantized model '{f_path}'. File not found.")
        try:
            with open(f_path, "rb") as f:
                pkl_data = pickle.load(f)
                def __extract_version_from_file(pkl_data):
                    # V1 does not contain an explicit field for version
                    return 1 if (not isinstance(pkl_data, dict)) else int(pkl_data["version"])
                __mera_ver = __extract_version_from_file(pkl_data)

                # Backwards compatible unpacking.
                if __mera_ver == 1: # (input_desc, ser_tvm_ir, q_params, ser_params, flow)
                    input_desc, ser_tvm_ir, q_params, ser_params = pkl_data[:4]
                elif __mera_ver == 2: # (input_desc, ser_tvm_ir, q_params, ser_params)
                    input_desc, ser_tvm_ir, q_params, ser_params = pkl_data["data"]
                else:
                    raise ValueError(f"Cannot load .mera models saved with a higher file version: {__mera_ver}.\n"
                        + "Please upgrade your version of MERA")
            from tvm.ir import load_json as __tvm_load_json
            tvm_ir = __tvm_load_json(ser_tvm_ir)
            return QuantizedMeraModelResult(input_desc, tvm_ir, q_params, ser_params, None)
        except Exception as ex:
            raise ValueError(f"Found error while loading MERA quantized model '{f_path}': {ex}")

    def measure_quantization_quality(self, dataset : Union[List[np.ndarray], List[List[np.ndarray]]],
            debug_mode : bool = False) -> MeraQualityContainer:
        if self.fp32_rtmod is None:
            raise ValueError(f'No fp32 reference model is available')
        node_list = list(self.fp32_rtmod._get_interpreter_node_list())
        node_data = {}

        _dataset = dataset if isinstance(dataset, list) else [dataset]
        qtz_out = []
        ref_out = []
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # Save model to temp dir
            model_path = tmpdir / "model.mera"
            self.save(model_path)
            # Deploy model
            with mera.TVMDeployer(tmpdir / "_out") as deployer:
                model = mera.ModelLoader(deployer).from_quantized_mera(model_path)
                deploy = deployer.deploy(model, target=Target.InterpreterHw)
                runner = deploy.get_runner()
            if debug_mode:
                qtzed_node_list = runner.rt_mod._get_interpreter_node_list()
            num_inputs = model.input_desc.num_inputs
            for inp in tqdm(_dataset, ncols=100, colour='yellow', desc='Evaluating quality', unit=' images'):
                if isinstance(inp, tuple):
                    inp = list(inp)
                elif not isinstance(inp, list):
                    inp = [inp]
                assert len(inp) == num_inputs, f"Input evaluation set must be a list of {num_inputs} elements (num inputs)."
                qtz_out.append(tuple(runner.set_input(inp).run().get_outputs()))
                [self.fp32_rtmod.set_input(i, inp[i]) for i in range(num_inputs)]
                # TODO - Use MERA interface
                self.fp32_rtmod.run()
                ref_out.append(tuple([self.fp32_rtmod.get_output(i).asnumpy() for i in range(self.fp32_rtmod.get_num_outputs())]))
                if debug_mode:
                    for op in node_list:
                        if op not in qtzed_node_list:
                            continue
                        if op not in node_data:
                            node_data[op] = {}
                            node_data[op]["ref"] = []
                            node_data[op]["got"] = []
                        node_data[op]["ref"].append(self.fp32_rtmod._get_interpreter_buffer(op))
                        node_data[op]["got"].append(runner.rt_mod._get_interpreter_buffer(op))
        out_qlty = calculate_quantization_quality(ref_out, qtz_out)
        if debug_mode:
            return MeraDebugQualityContainer(out_qlty, node_data, self.q_params)
        else:
            return MeraQualityContainer(out_qlty)


class ModelQuantizer:
    """Class to perform quantization of a model targeting MERA stack."""

    def __init__(self, model : Union[str, Path, TorchScriptModule, TorchModule, mera.MeraModel],
            input_shape : Union[Tuple, List[Tuple]],
            layout : Layout = Layout.NHWC):
        """Loads a model in which MERA can perform quantization for heterogeneous ML frameworks.

        :param model: Model to include. This can be set in many formats depending on the input framework:
          * PyTorch: Accepts a torchscript traced module, a path to a serialised traced torchscript module (*.pt extension) or a torch module.
          * ONNX: Accepts a path to a serialized ONNX model (*.onnx extension) whose op-set is one of the following: [10].
          * TFLite: Accepts a path to a serialized tensorflow lite model (*.tflite extension).
        :param input_shape: Specify a tuple with the shape of the input, or a list of tuples in case the model has multiple inputs.
        :param layout: Specify the 4D tensor layout of the model.
        """
        # Model Validation
        if isinstance(input_shape, tuple):
            input_shape = [input_shape]
        self.model, self.flow = ModelQuantizer.__model_load(model, input_shape)

        self.layout = Layout(layout)
        self.calibrated = False
        self.transformed = False

        if not input_shape:
            raise ArgumentError(f'input_shape argument needs to be provided.')
        self.input_shape = input_shape if isinstance(input_shape, list) else [input_shape]
        if not np.all([isinstance(x, tuple) for x in self.input_shape]):
            raise ArgumentError(f'All provided input shapes must be tuples')

        # Prepare quantizer mod
        logger.info(f'Loading and converting {str(self.flow.value)} model to MERA model...')
        if self.flow == ModelQuantizerFlow.TORCH:
            # FIXME - Hardcoded input to f32
            self.input_type = [np.float32] * len(self.input_shape)

            from tvm.relay.frontend import from_pytorch as __tvm_from_pytorch
            self.input_desc = [(f'input_{idx}', (tuple(shp), 'float32')) for idx,shp in enumerate(self.input_shape)]
            mod, params = __tvm_from_pytorch(self.model, self.input_desc, layout=self.layout.value)
        elif self.flow == ModelQuantizerFlow.ONNX:
            # FIXME - Hardcoded input to f32
            self.input_type = [np.float32] * len(self.input_shape)

            from tvm.relay.frontend import from_onnx as __tvm_from_onnx
            self.input_desc = [(f'input_{idx}', (tuple(shp), 'float32')) for idx,shp in enumerate(self.input_shape)]
            mod, params = __tvm_from_onnx(self.model)
        elif self.flow == ModelQuantizerFlow.TF_LITE:
            from tvm.relay.frontend import from_tflite as __tvm_from_tflite
            self.input_desc = {f'input_{idx}' : shp for idx,shp in enumerate(self.input_shape)}
            self.input_type = [np.float32] * len(self.input_shape)
            dtype_dict = {f'input_{x}' : 'float32' for x in range(len(self.input_shape))}
            mod, params = __tvm_from_tflite(self.model, shape_dict=self.input_desc, dtype_dict=dtype_dict)
        elif self.flow == ModelQuantizerFlow.MERA_MODEL:
            self.input_desc = self.model.input_desc
            self.input_shape = [inp.input_shape for inp in self.input_desc.all_inputs.values()]
            self.input_type = [desc.input_type for desc in self.input_desc.all_inputs.values()]
            mod, params = self.model._load_model_tvm()
        else:
            raise ValueError(f'Unsupported quantization flow {self.flow}')

        from tvm.relay.mera import build_fp32 as __build_fp32
        from tvm.relay.mera import build_config as __build_config
        with __build_config(target='Quantizer'):
            json, lib, params, mod = __build_fp32(mod, params, 'Quantizer', "x86")
        self.mod = mod
        self.params = params

        from tvm.runtime import cpu as __cpu
        from tvm.runtime import save_param_dict as __save_param_dict
        from tvm.contrib.graph_executor import create as __create
        self.qtzer_mod = __create(json, lib, __cpu())
        param_bytes = __save_param_dict(params)
        self.qtzer_mod.load_params(param_bytes)
        logger.info(f'Model ready for quantization')

    def __model_load(mod_input, in_shape):
        if isinstance(mod_input, mera.MeraModel):
            return mod_input, ModelQuantizerFlow.MERA_MODEL
        elif isinstance(mod_input, torch.jit.ScriptModule):
            return mod_input, ModelQuantizerFlow.TORCH
        elif isinstance(mod_input, str) or isinstance(mod_input, Path):
            # Treat it as a path
            model_path = Path(str(mod_input))

            if model_path.suffix == '.pt':
                if not model_path.exists():
                    raise ArgumentError(f'Could not find PyTorch traced model file {model_path}')
                with torch.no_grad():
                    return torch.jit.load(str(model_path)), ModelQuantizerFlow.TORCH
            elif model_path.suffix == '.onnx':
                if not model_path.exists():
                    raise ArgumentError(f'Could not find ONNX model file {model_path}')
                from .mera_model import MeraModelOnnx
                return MeraModelOnnx._resolve_model(model_path, 1), ModelQuantizerFlow.ONNX
            elif model_path.suffix == '.tflite':
                if not model_path.exists():
                    raise ArgumentError(f'Could not find TFLite model file {model_path}')
                from tflite import Model as __tfliteModel
                return __tfliteModel.GetRootAsModel(model_path.read_bytes(), 0), ModelQuantizerFlow.TF_LITE
            else:
                raise ArgumentError(f"Unknown or unsupported model file {model_path.suffix}.\n"\
                 + "Currently only PyTorch, ONNX or TFLite models supported for quantization")
        elif isinstance(mod_input, TorchModule):
            # Generate a script module out of it
            assert len(in_shape) == 1, "Automatic tracing with multiple inputs not implemented"
            with torch.no_grad():
                return torch.jit.trace(mod_input, torch.randn(*in_shape[0], dtype=torch.float32)).eval(), ModelQuantizerFlow.TORCH

    def __parse_q_params(self):
        q_params_str = self.qtzer_mod["mera_calculate_qparams"]()
        q_params = json.loads(q_params_str)
        all_params = {}
        for q_param in q_params:
            all_params = {**all_params, **q_param}
        return all_params

    def calibrate(self, calibration_inputs : Union[List[np.ndarray], List[List[np.ndarray]]]):
        """Feeds a series of realistic input data tensors which will calibrate the model for better quantization quality.
        MERA will observe the data ranges on all nodes across the whole model and will use that information to determine what
        are the best quantization domains based on the representative dataset and the user configuration. It is recommended to use
        a big enough dataset of realistic input data in order to obtain the best accuracy results for quantization.

        :param calibration_inputs: List of NumPy tensors with a representative dataset of real data. The length of this list will
        determine how many calibration iterations are performed. If the model contains more than one input, a list of lists is
        expected as input in the format of C[NC][NI], where NC is the number of calibration images and NI is the number of inputs.
        """
        c_ins = list(calibration_inputs)
        n_runs = int(len(c_ins))
        if n_runs == 0:
            raise ArgumentError(f'At least 1 calibration image needs to be provided')

        self.reset_calibration()
        logger.info(f'Running model quantization with {n_runs} images ...')
        for it in tqdm(range(n_runs), ncols=100, colour='yellow', desc='Running calibration', unit=' images',
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining},{rate_noinv_fmt}{postfix}]'):
            in_data = c_ins[it]
            if isinstance(in_data, tuple):
                calib_type = [x.dtype for x in in_data]
            else:
                if not isinstance(in_data, list):
                    in_data = [in_data]
                calib_type = [x.dtype for x in in_data]
            if self.input_type != calib_type:
                raise ValueError(f'Incorrect data type found for image #{it}: '
                 + f'Expected {self.input_type} but got {calib_type}')
            [self.qtzer_mod.set_input(i, d) for i,d in enumerate(in_data)]
            self.qtzer_mod.run()
        self.calibrated = True
        return self

    def reset_calibration(self):
        self.qtzer_mod["mera_quantizer_reset"]()
        self.calibrated = False
        return self

    def quantize(self) -> QuantizedMeraModelResult:
        if not self.calibrated:
            logger.warning(f'No calibration results are available for this model. Will use default Quantization params.\n'
            + 'Please use calibrate() method to compute quantization domains.')
        from tvm.relay.mera import qtz_transform as __mera_qtz_transform
        qtzed_mod, new_params = __mera_qtz_transform(self.mod, self.qtzer_mod)
        new_params = {**new_params, **self.params}
        return QuantizedMeraModelResult(self.input_desc, qtzed_mod, self.__parse_q_params(), new_params, self.qtzer_mod)
