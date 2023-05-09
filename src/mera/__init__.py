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
"""Mera: Public API for Mera ML compiler stack."""

from .version import __version__, get_versions, get_mera_version, get_mera_dna_version, get_mera_tvm_version

from .deploy import TVMDeployer

from .mera_model import ModelLoader, MeraModel

from .mera_deployment import load_mera_deployment, MeraTvmDeployment, MeraTvmPrjDeployment, MeraTvmModelRunner

from .deploy_project import Target, Layout

from .mera_platform import Platform

try:
    from .mera_quantizer import ModelQuantizer
except ModuleNotFoundError:
    pass


from .quantization_quality import QuantizationQualityMetrics, calculate_quantization_quality

from .model.input_desc import InputDescription, InputDescriptionContainer

from .metrics.power_metrics import PowerMetrics
