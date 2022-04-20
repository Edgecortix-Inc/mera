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
"""
Resnet50 PyTorch - Simulator
============================

This tutorial shows how to compile and run quantized PyTorch models using MERA compiler.
In this tutorial we use `resnet50` as an example and we deploy it targeting Simulator
on DNA `F200` architecture.
"""
##############################################################################
# Necessary imports and helpers
# -----------------------------
# We need to import the MERA environment. We also provide some helper functions
# to quantize a PyTorch model as well as converting an input tensor into NHWC format.
import torch
import numpy as np

import mera
from mera import Target, Platform

def quantize_model(model, inp):
    model.fuse_model()
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model, inplace=True)
    model(inp)
    torch.quantization.convert(model, inplace=True)


def nchw_to_nhwc(arr):
    if len(arr.shape) != 4:
        return arr
    N, C, H, W = arr.shape
    ret = np.zeros((N, H, W, C), dtype=arr.dtype)
    for n in range(N):
        for c in range(C):
            for h in range(H):
                for w in range(W):
                    ret[n, h, w, c] = arr[n, c, h, w]
    return ret


##############################################################################
# Specify input layout and configure deployment
# ---------------------------------------------
# Although PyTorch only supports NCHW layout, for performance reason it is
# often beneficial to take inputs in NHWC layout. For example, if images from
# raw sensors are in NHWC layout, converting them to NCHW layout to satisfy
# PyTorch's requirement would incur performance cost.
#
# MERA can take inputs in both NCHW and NHWC layouts, so you can choose
# either one depending on your input source. Internally, we always convert
# inputs to NHWC layout since that is what the DNA architecture expects, 
# so if the inputs are already in NHWC layout, there would be less layout 
# conversions.
#
# We can specify as well for Simulator target what is the clock frequency in MHz
# that we will use. This is used to report more accurate latency metrics after a
# run. In this example we set it to `300MHz` but could be changed based on the IP.
#
# We pick our DNA platform for deployment to be `DNAF200L0003` (F200), we could
# change this value based on the different DNA platforms installed in the system.
# Plase check `README.md` or documentation to see which are the other valid values.
torch.manual_seed(123)
layout = "NHWC"  # or NCHW

# Path to the traced model file as well as the mapping for input shape and type
model_path = 'resnet50.pt'
input_desc = {'input0' : ((1, 224, 224, 3), 'float32')}

# MERA platform to deploy for
mera_platform = Platform.DNAF200L0003
# Folder where we should create our MERA project
out_dir = "deployment_resnet50"
inp = torch.rand((1, 3, 224, 224))

SIM_FREQ = 300
build_cfg = {
    'sim_freq_mhz' : SIM_FREQ
}

################################################################################
# Quantize and trace resnet50 PyTorch Model
def prepare_model(input_data):
    from torchvision.models.quantization import resnet as qresnet
    model = qresnet.resnet50(pretrained=True).eval()
    quantize_model(model, input_data)
    with torch.no_grad():
        script_module = torch.jit.trace(model, input_data).eval()
    torch.jit.save(script_module, model_path)
    print(f'resnet50 model quantized and saved to {model_path}')
prepare_model(inp)

################################################################################
# Here, we specify which MERA "target" we want to compile for.
# We have the following valid targets: `IP`, `Simulator` and `VerilatorSimulator`:
# * IP deploys a model intended to be run on a DNA accelerator in HW. It can target
#   x86 or arm (cross-compile) host_arch.
# * Simulator deploys a model which can be run using the host simulator provided
#   by MERA.
# * VerilatorSimulator deploys a model that will be run using a host hw emulator.
target = Target.Simulator


################################################################################
# Deploy and run for Simulator
# ----------------------------
# We load and convert the Pytorch model into a MERA compatible model.
# Calling from_pytorch() accepts the traced script module saved into a file.
# We need to provide the input description map with the shapes and types of the
# input variables.
# The model_name will be picked from the file_name by default unless overriden
# 
# deploy(...) is the main entry point to the MERA compiler. It runs the backend
# and outputs compile artifacts in the output directory specified in the
# TVMDeployer. We require the model loaded as a mera compatible model passed to
# deploy(), the other arguments contain defaults but we also want to provide the
# value for target to specify we are building for Simulator
#
# Besides the model and its parameters, another important argument
# is `host_arch`. `host_arch` should be "x86" or "arm". 
# For "InterpreterHw" and "Simulator" targets, `host_arch` must be "x86".
#
# The output may contain several warnings regarding untyped tensors. These
# can be safely ignored as they don't affect the MERA framework.
with mera.TVMDeployer(out_dir, overwrite=True) as deployer:
    model = mera.ModelLoader(deployer).from_pytorch(model_path, input_desc)

    # Run on Interpreter
    print(f'Deploying resnet50 for InterpreterHw...')
    deploy_int = deployer.deploy(model, mera_platform=mera_platform, build_config=build_cfg, target=Target.InterpreterHw)
    # If the input layout is NHWC, it is important not to forget to convert the
    # input given to PyTorch to NHWC.
    nhwc_inp = inp.numpy()
    if layout == "NHWC":
        nhwc_inp = nchw_to_nhwc(inp.numpy())
    int_result = deploy_int.get_runner().set_input(nhwc_inp).run().get_outputs()
    # Save output data files for later comparison
    for idx, res in enumerate(int_result):
        deployer.save_data_file(f'ref_result_{idx}.bin', res.flatten().astype(np.float32))

    # Compile for simulator
    print(f'Deploying resnet50 for {target}...')
    deploy = deployer.deploy(model, mera_platform=mera_platform, build_config=build_cfg, target=target, host_arch='x86')

################################################################################
# Run on target and check results.
run_res = deploy.get_runner().set_input(nhwc_inp).run()

print(f'Checking output data against InterpreterHw...')
for int_data, hw_data in zip(int_result, run_res.get_outputs()):
    assert np.allclose(int_data, hw_data)

##########################################################################
# Obtain the elapsed latency
# --------------------------
# `get_runtime_metrics()` method on a runner module can be used to query
# the elapsed latency and other metrics. For "Simulator" target, this is a 
# latency estimate summed over all functions executed on the simulator.
# These values are stored in the `sim_time_us` metric, so collect them all to
# report to the user.
total_latency = sum(x.get('sim_time_us', 0) for x in run_res.get_runtime_metrics()) / 1000
print(f"Simulator total latency ({SIM_FREQ} MHz): {total_latency} ms")

print(f'SUCCESS')
