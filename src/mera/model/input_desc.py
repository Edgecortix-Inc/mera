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

import numpy as np

from typing import Tuple

class InputDescription:
    """Class representing information about a model's input."""

    def __init__(self, input_name : str, input_shape : Tuple[int], input_type = 'float32'):
        self.__input_name = input_name
        self.__input_shape = input_shape
        if isinstance(input_type, np.dtype):
            input_type = str(input_type.name)
        if isinstance(input_type, str):
            __valid_str_types = ['float32', 'double', 'float64', 'int32', 'int8', 'uint8', 'int64']
            if input_type not in __valid_str_types:
                raise ValueError(f"Input type {input_type} is not one of the valid strings: {__valid_str_types}")
            self.__input_type = input_type
        else:
            raise ValueError(f"Could not parse input type information {input_type} [{type(input_type)}]")

    @property
    def input_name(self):
        return self.__input_name

    @property
    def input_shape(self):
        return self.__input_shape

    @property
    def input_type(self):
        return self.__input_type


class InputDescriptionContainer:
    """Class representing information about a collection of model inputs."""

    def __init__(self, data):
        self.all_inputs = dict()
        if isinstance(data, InputDescriptionContainer):
            self.all_inputs = data.all_inputs
        elif isinstance(data, InputDescription):
            self.all_inputs = {data.input_name : data}
        elif isinstance(data, list):
            for d in data:
                if isinstance(d, InputDescription):
                    self.all_inputs[d.input_name] = d
                elif isinstance(d, tuple):
                    assert len(d) == 2 and isinstance(d[1], tuple) and len(d[1]) == 2, \
                        f"Could not parse input description from tuple: {d}"
                    in_name, in_data = d
                    self.all_inputs[in_name] = InputDescription(in_name, in_data[0], in_data[1])
        elif isinstance(data, dict):
            for in_name, in_data in data.items():
                assert isinstance(in_data, tuple) and len(in_data) == 2, \
                    f"Could not parse tuple of (input_shape, input_type) from data on input {in_name} : '{in_data}'"
                self.all_inputs[in_name] = InputDescription(in_name, in_data[0], in_data[1])
        else:
            raise ValueError(f'Could not parse Input Description information out of {data}')

    def _join(others):
        assert isinstance(others, list)
        res = InputDescriptionContainer({})
        for idx, cntr in enumerate([InputDescriptionContainer(x) for x in others]):
            for in_name, in_desc in cntr.all_inputs.items():
                res.all_inputs[in_name + "_" + str(idx)] = in_desc
        return res

    def to_dict(self):
        return {in_name : (in_desc.input_shape, in_desc.input_type) for in_name,in_desc in self.all_inputs.items()}

    def _for_tvm_pytorch(self):
        # List of tuple with (name, desc); where desc is tuple of (shape, dtype)
        return [(in_name, (in_data.input_shape, in_data.input_type)) for in_name,in_data in self.all_inputs.items()]

    def _for_tvm_tflite(self):
        # Map of name->shape and map of name->dtype
        return {in_name : in_data.input_shape for in_name,in_data in self.all_inputs.items()}, \
            {in_name : in_data.input_type for in_name,in_data in self.all_inputs.items()}

    @property
    def num_inputs(self):
        return len(self.all_inputs)
