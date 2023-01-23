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
Fused Resnet18 + MobilenetV2 PyTorch - Simulator
================================================

This tutorial shows how to fuse two quantized PyTorch models and then complie and run the fused model using MERA compiler.
Specifically, we fuse `resnet18` and `mobilenet_v2` targeting Simulator on DNA `F200` architecture.
Fusing multiple models is especially useful for fully utilizing the compute resources of a large platform.
"""
##############################################################################
# Necessary imports and helpers
# -----------------------------
# We need to import the MERA environment. We also provide some helper functions
# to quantize a PyTorch model as well as converting an input tensor into NHWC format.
import torch, torchvision
import numpy as np

import mera
from mera import Target, Platform, Layout

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
layout = Layout.NHWC # or Layout.NCHW
input_desc = {'input0' : ((1, 224, 224, 3), 'float32')}

target = Target.Simulator
mera_platform = Platform.DNAA600L0002

SIM_FREQ = 1000
build_cfg = {'sim_freq_mhz' : SIM_FREQ}


################################################################################
# Quantize and trace resnet18 PyTorch Model
torch.manual_seed(123)
inp = torch.rand((1, 3, 224, 224))

resnet18 = torchvision.models.quantization.resnet18(pretrained=True).eval()
quantize_model(resnet18, inp)
with torch.no_grad():
    script_module = torch.jit.trace(resnet18, inp).eval()
torch.jit.save(script_module, "resnet18.pt")
print('resnet18 quantized and saved to resnet18.pt')

# Quantize and trace mobilenet_v2 PyTorch Model
mobilenet_v2 = torchvision.models.quantization.mobilenet_v2(pretrained=True).eval()
quantize_model(mobilenet_v2, inp)
with torch.no_grad():
    script_module = torch.jit.trace(mobilenet_v2, inp).eval()
torch.jit.save(script_module, "mobilenet_v2.pt")
print('mobilenet_v2 quantized and saved to mobilenet_v2.pt')


################################################################################
# Fuse resnet18 + mobilenet_v2 and deploy for Simulator
# -----------------------------------------------------
# We first load the resnet Pytorch model and the mobilenet Pytorch model separately by calling from_pytorch().
# Then we fuse the two models by calling fuse_models(), which returns a single fused model for compilation and deployment.
# To make resnet and mobilenet share the same input, we set share_input to True when calling fuse_models().
out_dir = "deployment_fused_resnet_mobilenet"
with mera.TVMDeployer(out_dir, overwrite=True) as deployer:
    resnet = mera.ModelLoader(deployer).from_pytorch("resnet18.pt", input_desc, layout=layout)
    mobilenet = mera.ModelLoader(deployer).from_pytorch("mobilenet_v2.pt", input_desc, layout=layout)
    fused_model = mera.ModelLoader(deployer).fuse_models([resnet, mobilenet], share_input=True)

    # Run on Interpreter
    print(f'Deploying fused resnet18 + mobilenet_v2 for InterpreterHw...')
    deploy_int = deployer.deploy(fused_model, mera_platform=mera_platform, build_config=build_cfg, target=Target.InterpreterHw)
    nhwc_inp = inp.numpy()
    if layout == Layout.NHWC:
        nhwc_inp = nchw_to_nhwc(inp.numpy())
    int_run = deploy_int.get_runner().set_input(nhwc_inp).run()
    # Get the first output of the fused model, which is the output of resnet
    int_result_resnet = int_run.get_output(0)
    # Get the second output of the fused model, which is the output of mobilenet
    int_result_mobilenet = int_run.get_output(1)

    # Run for Simulator
    print(f'Deploying fused resnet18 + mobilenet_v2 for Simulator...')
    deploy_sim = deployer.deploy(fused_model, mera_platform=mera_platform, build_config=build_cfg, target=target, host_arch='x86')
    sim_run = deploy_sim.get_runner().set_input(nhwc_inp).run()
    sim_result_resnet = sim_run.get_output(0)
    sim_result_mobilenet = sim_run.get_output(1)

    # Check correctness
    print(f'Checking Simulator results against InterpreterHw...')
    assert np.allclose(int_result_resnet, sim_result_resnet)
    assert np.allclose(int_result_mobilenet, sim_result_mobilenet)
    print(f'SUCCESS')
