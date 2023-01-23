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
import logging
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

class QuantizedMeraModelResult:
    """Class that represents the result of a model quantized with the MERA quantizer."""

    def __init__(self, input_desc, qtzed_mod, q_params, params, orig_model, flow, fp32_rtmod):
        self.input_desc = input_desc
        self.qtzed_mod = qtzed_mod
        self.q_params = q_params
        self.params = params
        self.orig_model = orig_model
        self.flow = flow
        self.fp32_rtmod = fp32_rtmod

    def save(self, file_name : Union[str, Path]) -> None:
        f_path = Path(file_name).resolve()
        logger.debug(f"Saving quantized MERA model to '{f_path}' ...")
        from tvm.ir import save_json as __tvm_save_json
        ser_tvm_ir = __tvm_save_json(self.qtzed_mod)
        ser_params = {k: v.numpy() for k,v in self.params.items()}
        pickle_data = (self.input_desc, ser_tvm_ir, self.q_params, ser_params, self.flow)
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
                input_desc, ser_tvm_ir, q_params, ser_params, flow = pickle.load(f)
            from tvm.ir import load_json as __tvm_load_json
            tvm_ir = __tvm_load_json(ser_tvm_ir)
            return QuantizedMeraModelResult(input_desc, tvm_ir, q_params, ser_params, None, flow, None)
        except Exception as ex:
            raise ValueError(f"Found error while loading MERA quantized model '{f_path}': {ex}")

    def __get_ref_data(self, model, np_data):
        if self.flow == ModelQuantizerFlow.TORCH:
            with torch.no_grad():
                out_pt = model(torch.from_numpy(np_data))
                out_flatten = []
                if isinstance(out_pt, tuple) or isinstance(out_pt, list):
                    for x in out_pt:
                        if isinstance(x, list):
                            out_flatten += [xx.numpy() for xx in x]
                        else:
                            out_flatten.append(x.detach().numpy())
                    return tuple(out_flatten)
                else:
                    return tuple(out_pt.detach().numpy())
        elif self.flow == ModelQuantizerFlow.ONNX:
            import onnxruntime
            session = onnxruntime.InferenceSession(model, None)
            if len(session.get_inputs()) > 1:
                session_data = {s.name:val for s,val in zip(session.get_inputs(), np_data)}
            else:
                session_data = {session.get_inputs()[0].name : np_data}
            return tuple(session.run(None, session_data))
        else:
            raise ValueError(f'Unsupported quantization runtime flow {self.flow}')

    def measure_quantization_quality(self, dataset : Union[List[np.ndarray], List[List[np.ndarray]]],
        model : Union[TorchModule, TorchScriptModule] = None, debug_mode : bool = False) -> MeraQualityContainer:
        from .deploy import TVMDeployer
        from .mera_model import ModelLoader
        _model = model if model else self.orig_model
        assert _model is not None, "No external model is available for the quality measurement"

        # TODO - Just use fp32_mod instead of raw_model
        if debug_mode:
            if self.fp32_rtmod is None:
                raise ValueError(f'No fp32 reference model is available')
            node_list = self.fp32_rtmod._get_interpreter_node_list()
            node_data = {}

        _dataset = dataset if isinstance(dataset, list) else [dataset]
        with TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            # Save model to temp dir
            model_path = tmpdir / "model.mera"
            self.save(model_path)
            if self.flow == ModelQuantizerFlow.ONNX:
                import onnx
                onnx_path = str(tmpdir / "ref_model.onnx")
                onnx.save(_model, onnx_path)
                _model = onnx_path
            # Deploy model
            with TVMDeployer(tmpdir / "_out") as deployer:
                model = ModelLoader(deployer).from_quantized_mera(model_path)
                deploy = deployer.deploy(model, target=Target.InterpreterHw)
                runner = deploy.get_runner()
            if debug_mode:
                qtzed_node_list = runner.rt_mod._get_interpreter_node_list()
            qtz_out = []
            ref_out = []
            for inp in tqdm(_dataset, ncols=100, colour='yellow', desc='Evaluating quality', unit=' images'):
                if isinstance(inp, tuple):
                    inp = list(inp)
                qtz_out.append(tuple(runner.set_input(inp).run().get_outputs()))
                ref_out.append(self.__get_ref_data(_model, inp))
                if debug_mode:
                    self.fp32_rtmod.set_input(0, inp)
                    self.fp32_rtmod.run()
                    for op in list(node_list):
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

    def __init__(self, model : Union[str, Path, TorchScriptModule, TorchModule],
            input_shape : Union[Tuple, List[Tuple]] = None,
            layout : Layout = Layout.NHWC):
        """Loads a model in which MERA can perform quantization for heterogeneous ML frameworks.

        :param model: Model to include. This can be set in many formats depending on the input framework:
          * PyTorch: Accepts a torchscript traced module, a path to a serialised traced torchscript module (*.pt extension) or a torch module.
          * ONNX: Accepts a path to a serialized ONNX model (*.onnx extension) whose op-set is one of the following: [10].
          * TFLite: Accepts a path to a serialized tensorflow lite model (*.tflite extension).
        :param input_shape: When using input models with PyTorch, specify a tuple with the shape of the input,
            or a list of tuples in case the model has multiple inputs. Leave as None when using other model frameworks.
        :param layout: Specify the 4D tensor layout of the model.
        """
        # Model Validation
        if isinstance(input_shape, tuple):
            input_shape = [input_shape]
        self.model, self.flow = ModelQuantizer.__model_load(model, input_shape)

        self.layout = Layout(layout)
        self.calibrated = False
        self.transformed = False

        # Prepare quantizer mod
        logger.info(f'Loading and converting {str(self.flow.value)} model to MERA model...')
        if self.flow == ModelQuantizerFlow.TORCH:
            if not input_shape:
                raise ArgumentError(f'input_shape argument needs to be provided when using PyTorch quantization')
            self.input_shape = input_shape if isinstance(input_shape, list) else [input_shape]
            if not np.all([isinstance(x, tuple) for x in self.input_shape]):
                raise ArgumentError(f'All provided input shapes must be tuples')
            # FIXME - Hardcoded input to f32
            self.input_type = [np.float32] * len(self.input_shape)

            from tvm.relay.frontend import from_pytorch as __tvm_from_pytorch
            self.input_desc = [(f'input_{idx}', (tuple(shp), 'float32')) for idx,shp in enumerate(self.input_shape)]
            mod, params = __tvm_from_pytorch(self.model, self.input_desc, layout=self.layout.value)
        elif self.flow == ModelQuantizerFlow.ONNX:
            from tvm.relay.frontend import from_onnx as __tvm_from_onnx
            from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
            # Parse input_desc from ONNX model info
            self.input_desc = {}
            self.input_shape = []
            self.input_type = []
            for in_info in self.model.graph.input:
                self.input_type.append(TENSOR_TYPE_TO_NP_TYPE[int(in_info.type.tensor_type.elem_type)])
                shape_val = tuple([x.dim_value for x in in_info.type.tensor_type.shape.dim])
                self.input_desc[in_info.name] = shape_val
                self.input_shape.append(shape_val)
            mod, params = __tvm_from_onnx(self.model)
        elif self.flow == ModelQuantizerFlow.TF_LITE:
            from tvm.relay.frontend import from_tflite as __tvm_from_tflite
            self.input_desc = {f'input_{idx}' : shp for idx,shp in enumerate(self.input_shape)}
            self.input_type = [np.float32] * len(self.input_shape)
            dtype_dict = {f'input_{x}' : 'float32' for x in range(len(self.input_shape))}
            mod, params = __tvm_from_tflite(self.model, shape_dict=self.input_desc, dtype_dict=dtype_dict)
        else:
            raise ValueError(f'Unsupported quantization flow {self.flow}')

        from tvm.relay.mera import build_fp32 as __build_fp32
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
        if isinstance(mod_input, torch.jit.ScriptModule):
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
                import onnx
                model = onnx.load(model_path)
                # Resolve symbolic batch num to 1 for quantization
                for input in model.graph.input:
                    n_dim = input.type.tensor_type.shape.dim[0]
                    if n_dim.dim_param == 'N':
                        # Change symbolic batch with 1
                        n_dim.dim_value = 1
                return model, ModelQuantizerFlow.ONNX
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
                calib_shape = [x.shape for x in in_data]
                calib_type = [x.dtype for x in in_data]
            else:
                if not isinstance(in_data, list):
                    in_data = [in_data]
                calib_shape = [x.shape for x in in_data]
                calib_type = [x.dtype for x in in_data]
            if self.input_shape != calib_shape:
                raise ValueError(f'Incorrect data shape found for image #{it}: '
                 + f'Expected {self.input_shape} but got {calib_shape}')
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
        return QuantizedMeraModelResult(self.input_desc, qtzed_mod, self.__parse_q_params(), new_params,
            self.model, self.flow, self.qtzer_mod)
