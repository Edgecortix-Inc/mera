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
EfficientNet Lite - IP
======================

 This tutorial shows how to compile and run quantized TFLite models (EfficientNet) using MERA.
 We deploy and run them targeting `IP` accelerator and DNA `F200` architecture.
 
 Model details: https://arxiv.org/abs/1905.11946
 
 * CPU/GPU/TPU latency are measured on Pixel4, with batch size 1 and 4 CPU threads. FP16 GPU latency is measured with default latency, while FP32 GPU latency is measured with additional option `--gpu_precision_loss_allowed=false`.

+--------------------+------------+-----------+-------------------+-----------------------+----------------------+----------------------+-------------------+-----------------------+----------------------+
| **Model**          | **params** | **MAdds** | **FP32 accuracy** | **FP32 CPU  latency** | **FP32 GPU latency** | **FP16 GPU latency** | **INT8 accuracy** | **INT8 CPU latency**  | **INT8 TPU latency** |
+====================+============+===========+===================+=======================+======================+======================+===================+=======================+======================+
| efficientnet-lite0 | 4.7M       | 407M      |  75.1%            |  12ms                 | 9.0ms                | 6.0ms                | 74.4%             |  6.5ms                | 3.8ms                |
+--------------------+------------+-----------+-------------------+-----------------------+----------------------+----------------------+-------------------+-----------------------+----------------------+
| efficientnet-lite1 | 5.4M       | 631M      |  76.7%            |  18ms                 | 12ms                 | 8.0ms                |  75.9%            | 9.1ms                 | 5.4ms                |
+--------------------+------------+-----------+-------------------+-----------------------+----------------------+----------------------+-------------------+-----------------------+----------------------+
| efficientnet-lite2 | 6.1M       | 899M      |  77.6%            |  26ms                 | 16ms                 | 10ms                 | 77.0%             | 12ms                  | 7.9ms                |
+--------------------+------------+-----------+-------------------+-----------------------+----------------------+----------------------+-------------------+-----------------------+----------------------+
| efficientnet-lite3 | 8.2M       | 1.44B     |  79.8%            |  41ms                 | 23ms                 | 14ms                 | 79.0%             | 18ms                  | 9.7ms                |
+--------------------+------------+-----------+-------------------+-----------------------+----------------------+----------------------+-------------------+-----------------------+----------------------+
| efficientnet-lite4 | 13.0M      | 2.64B     |  81.5%            |  76ms                 | 36ms                 | 21ms                 | 80.2%             | 30ms                  |                      |
+--------------------+------------+-----------+-------------------+-----------------------+----------------------+----------------------+-------------------+-----------------------+----------------------+
 
 Original repository: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/lite
"""

###############################################################################
# Necessary imports and Helpers
# -----------------------------
# We need to import the MERA environment and we provide a helper function
# to load and crop an input image and another helper function to download and
# extract the `EfficientNet` int8 model file from the internet and save it to
# a local folder.
import numpy as np
import tensorflow as tf
import os

import mera
from mera import Target
from mera import Platform

# Load image helper
def load_image(image_path, input_size):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    input_image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)
    input_image = tf.cast(input_image, tf.uint8)
    return input_image

# Utility class for downloading EfficientNet model files
def download_models(dst):
    import subprocess, tempfile, tarfile
    from pathlib import Path

    _data_dir = Path(dst)
    print('Downloading EfficientNet models...')

    def _get_model(url, dst):
        _file_name = Path(url).name
        with tempfile.TemporaryDirectory() as tmpdir:
            p = subprocess.Popen(f'cd {tmpdir} && wget {str(url)}', stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell = True)
            p.communicate()
            with tarfile.open(Path(tmpdir) / _file_name) as tar_data:
                # Remove '.tar.gz' from filename
                model_name = _file_name[:-len('.tar.gz')]
                model = tar_data.extractfile(f'{model_name}/{model_name}-int8.tflite').read()
            with open(dst, 'wb') as w:
                w.write(model)
    _get_model('https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite1.tar.gz',
        _data_dir / 'effnet-lite1.tflite')
    _get_model('https://storage.googleapis.com/cloud-tpu-checkpoints/efficientnet/lite/efficientnet-lite4.tar.gz',
        _data_dir / 'effnet-lite4.tflite')


##########################################################################
# Running MERA Compilation
# ------------------------
# First of all create a MERA deployment project under 'deploy_effnet_liteX'
# We set the overwrite flag to true to clean the directory for every run
#
# Next, convert the input model into a MERA compatible model. Calling `from_tflite()`
# accepts the TFLite model saved into a file.
# The model_name will be picked from the file_name by default unless overridden
# with model_name
#
# Lastly deploy the model, targeting "IP" target.
# We can specify `host_arch` to be "arm", to cross-compile the host code
# to run on a 64-bit ARM host, or "x86" depending on the architecture
# where the accelerator lives.
#
# The compiled artifacts will be stored in `<output_dir>/result/`.
def mera_compile(tflite_filename, image_path, platform, host_arch, output_dir):
    print(f'\nDeploying MERA model "{tflite_filename}" ...')
    with mera.TVMDeployer(output_dir, overwrite=True) as deployer:
        model = mera.ModelLoader(deployer).from_tflite(tflite_filename)
        # Grab the 'height' component (NHWC)
        input_size = model.get_input_shape()[1]
        # Load the input data
        input_data = np.array(load_image(image_path, input_size))
        deployer.deploy(model, mera_platform=platform, target=Target.IP, host_arch=host_arch)
    return input_data


###############################################################################
# Configure MERA Deployments
# --------------------------
# As a first stage we download from the internet the EfficientNet model file.
# We then launch deployments targetting `IP` for the `DNAF200L0003` MERA platform
# with their own deployment directory. Our host arch is `x86`, but we could choose
# `arm` for cross-compilation.
#
# The value for the DNA platform can be changed based on which platform is installed in the system.
# Plase check `README.md` or HTML documentation to see which are the other valid values.
#
# The `mera_compile()` script will also resize the input image to fit the model's resolution
download_models('data')

# The raw input data that we will use
image_path = 'data/cat.png'

# Output MERA project folders for EfficientNet lite 1 and 4
output_dir_lite1 = "deploy_effnet_lite1"
output_dir_lite4 = "deploy_effnet_lite4"

# The MERA platform that we build for
platform = Platform.DNAF200L0003
host_arch = "x86"

input_data_eflite1 = mera_compile("data/effnet-lite1.tflite", image_path, platform, host_arch, output_dir_lite1)
input_data_eflite4 = mera_compile("data/effnet-lite4.tflite", image_path, platform, host_arch, output_dir_lite4)


###############################################################################
# Load MERA deployment
# --------------------
# We don't need to redeploy and recompile the project if we already have one available
# (we built it in the previous stage with `mera_compile()`). So we use the utility function
# load_mera_deployment() which will detect a MERA project and fetch the necessary files
# to run a new simulation.
ip_lite1 = mera.load_mera_deployment(output_dir_lite1)
ip_lite4 = mera.load_mera_deployment(output_dir_lite4)




###############################################################################
# Run inference on IP accelerator
# -------------------------------
# After we have a deployment loaded for both models, we provide the scaled input data
# and run inference within the model. Afterwards we get the output variable.
#
# The runtime setup in the accelerator machine needs to be done before running this tutorial
# in order to successfully run in IP. For this tutorial as well please define `RUN_IP` env
# before running (this is meant to prevent failures when not running on an machine with a valid
# accelerator).
#
# From the runner we can query the runtime metrics which will contain different
# metrics about the simulation that has run. From that object we get the `elapsed_latency`
# metric and add all of them together (we might have multiple measurement points depending
# on the model). Given the metric units are in `us` we divide by 1000 to get `ms`.

# Toggle this variable to enable running on IP
RUN_ON_IP = os.environ.get('RUN_IP', False)

# Set EC_PROFILING env in order to collect latency metrics (for IP target only)
os.environ['EC_PROFILING'] = '1'

def get_total_latency_ms(run_result):
    metrics = run_result.get_runtime_metrics()
    total_us = sum([x.get('elapsed_latency', 0) for x in metrics])
    return total_us / 1000

if RUN_ON_IP:
    mera_runner_lite1 = ip_lite1.get_runner().set_input(input_data_eflite1).run()
    mera_result_lite1 = mera_runner_lite1.get_outputs()
    print("Optimized inference latency efficient net lite 1 (IP):", get_total_latency_ms(mera_runner_lite1), "ms")

    mera_runner_lite4 = ip_lite4.get_runner().set_input(input_data_eflite4).run()
    mera_result_lite4 = mera_runner_lite4.get_outputs()
    print("Optimized inference latency efficient net lite 4 (IP):", get_total_latency_ms(mera_runner_lite4), "ms")


###############################################################################
# Check Imagenet Results
# ----------------------
# Download the label set and compare the top 3 labels of both deployments.
# The expected result should be a classification of a cat.
from tvm.contrib.download import download_testdata
def get_synset():
    synset_url = "".join(
        [
            "https://gist.githubusercontent.com/zhreshold/",
            "4d0b62f3d01426887599d4f7ede23ee5/raw/",
            "596b27d23537e5a1b5751d2b0481ef172f58b539/",
            "imagenet1000_clsid_to_human.txt",
        ]
    )
    synset_name = "imagenet1000_clsid_to_human.txt"
    synset_path = download_testdata(synset_url, synset_name, module="data")
    with open(synset_path) as f:
        return eval(f.read())
if RUN_ON_IP:
    synset = get_synset()

    mera_top3_labels_lite1 = np.argsort(mera_result_lite1[0][0])[::-1][:3]
    mera_top3_labels_lite4 = np.argsort(mera_result_lite4[0][0])[::-1][:3]
    print("MERA compiled top3 labels lite 1:", [synset[label] for label in mera_top3_labels_lite1])
    print("MERA compiled top3 labels lite 4:", [synset[label] for label in mera_top3_labels_lite4])
