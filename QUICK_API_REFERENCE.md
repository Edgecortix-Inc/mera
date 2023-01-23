# Quick API reference

Below is a list of code snippets on how to interact with the MERA framework from the Python API.

## Creating a MERA deployment project

A MERA deployment project consists on a series of files and directories, all managed by MERA,
which allow the user to deploy and load models for different target of the MERA IP stack.

```python
# Base import
import mera

# Folder name relative to cwd where MERA will write all files for model deployments
DEPLOY_DIR = "my_deployment/"
# Whether the above DEPLOY_DIR should be cleaned up before any deployment, defaults to False
OVERWRITE_PROJECT = True

# With the following statement a new MERA project directory will be created and can be used
# by the `deployer` variable. You also acquire a directory lock on that folder for all statements
# inside the with block, so no 2 projects can deploy to the same place at once.
with mera.TVMDeployer(DEPLOY_DIR, overwrite=OVERWRITE_PROJECT) as deployer:
    ...
```

## Importing models into MERA

After a MERA deployment project is created you can import different quantized models to it by using the
`mera.ModelLoader` class. A `ModelLoader` instance can be created by passing to which deployment project
it should import the model and then it provides several methods for actually importing the model based on the different source frameworks.

You need to import your models with this `ModelLoader` in order to be able to deploy them with MERA.

### PyTorch pre-quantized models

Import PyTorch traced models in torchscript format using the `from_pytorch` method.

```python
import mera

MODEL_PATH = "./model.pt"
MODEL_INPUT_SHAPE = (1, 224, 224, 3)
MODEL_INPUT_TYPE = "float32"
# A map which contains each model's input name with a tuple of (input_shape, input_type)
INPUT_DESCRIPTION = {"input0": (MODEL_INPUT_SHAPE, MODEL_INPUT_TYPE)}

DEPLOY_DIR = "./model_deployment"
with mera.TVMDeployer(DEPLOY_DIR) as deployer:
    model = mera.ModelLoader(deployer).from_pytorch(MODEL_PATH, INPUT_DESCRIPTION)
    ...
```

### TFLite pre-quantized models

Import Tensorflow models in TFLite format using the `from_tflite` method.

```python
import mera

MODEL_PATH = "./model.tflite"

DEPLOY_DIR = "./model_deployment"
with mera.TVMDeployer(DEPLOY_DIR) as deployer:
    model = mera.ModelLoader(deployer).from_tflite(MODEL_PATH)
    ...
```

### MERA pre-quantized models

Import quantized MERA models in MERA format using `from_quantized_mera` method. See MERA Quantization reference for more information.

```python
import mera

MODEL_PATH = "./model.mera"

DEPLOY_DIR = "./model_deployment"
with mera.TVMDeployer(DEPLOY_DIR) as deployer:
    model = mera.ModelLoader(deployer).from_quantized_mera(MODEL_PATH)
    ...
```

### Fusing multiple models
To fully utilize the compute resources of a large platform, it is desirable to fuse multiple models into a single model for compilation and deployment.
With MERA, users can do model fusion using the `fuse_models` method.
Currently, it requires that the models to be fused should be from the same frontend, either Pytorch, Tflite or MERA.

```python
import mera

DEPLOY_DIR = "./model_deployment"
with mera.TVMDeployer(DEPLOY_DIR) as deployer:
    model_a = mera.ModelLoader(deployer).from_tflite("./model_a.tflite")
    model_b = mera.ModelLoader(deployer).from_tflite("./model_b.tflite")
    # Input sharing is supported when each model has exactly one input
    fused_model = mera.ModelLoader(deployer).fuse_models([model_a, model_b], share_input=False)
    ...
```

## MERA Quantization

Using the MERA quantizer premium feature you can take float32 models in PyTorch, TFLite or ONNX format
and quantized them natively using the MERA software environment. You will get a new quantized model format (`.mera`) that you can then use to deploy and run inference like any other model using MERA.

### Import float32 model

Create a `mera.ModelQuantizer` instance with the information about the float32 model you wish to quantize:

```python
import mera

#
# API reference:
# mera.ModelQuantizer(model, input_shape=None, layout=mera.Layout.NHWC)
#
#  * model: Can be a str/pathlib.Path representing the location of a PyTorch/TFLite/ONNX model file
#           in the disk, or the actual model object of any of those frameworks.
#  * input_shape: Only needed for PyTorch models. Provide a tuple with the shape of the input tensor.
#  * layout: If the model is not on NHWC layout, provide the model layout using the `mera.Layout` enum.

# Example ONNX/TFLite
quantizer = mera.ModelQuantizer("model.onnx") # or "model.tflite"

...

# Example PyTorch (with different layout)
INPUT_SHAPE = (1, 3, 224, 224)
quantizer = mera.ModelQuantizer("model.pt", input_shape=INPUT_SHAPE, layout=mera.Layout.NCHW)
```

### Calibration and Quantization

For good calibration you will need to provide a realistic data set. On these snippets calibration with
random data will be performed as an example.

```python
import mera
import numpy as np

# Create a MERA Quantizer instance by loading the float32 model.
mera_quantizer = mera.ModelQuantizer("model_fp32.onnx")

# Here you should generate a list of real data, showcasing random dataset of 30 images.
calibration_data = [np.random.randn(1, 224, 224, 3).astype(np.float32) for _ in range(30)]

# Using the calibrate() method, feed the calibration dataset.
# Then just call quantize and a quantized model will be generated.
model_qtz = mera_quantizer.calibrate(calibration_data).quantize()

# Lastly save your MERA quantized model to disk so that you can later import it with ModelLoader and deploy it.
model_qtz.save("model_int.mera")
```

### Quantization Quality

You can optionally generate a series of metrics in order to evaluate how good the quantized model is
vs the original float32 model. A bad quality can indicate errors or unsuitability for quantization.

You can compute the quality object using the `measure_quantization_quality` method on a quantized model.
This will give you a set of different metrics comparing the output from the float32 model against
the quantized model. Some useful global summary metrics are the following:
* **psnr**: PSNR of the f32 model output vs the int output. Values are on a logarithmic scale and
generally can be interpreted as such:
  * < 20: Poor quality.
  * 20-25: Acceptable quality.
  * 25-30: Good quality.
  * \>30: Very good quality.
* **score**: Measure of ratio of max abs error vs data range (max-min), normalised as a percentage.
Best possible value would be 100. With normally distributed output data, values < 75
can be considered poor, but depends heavely on the distribution of the output data. Use it in conjunction with the PSNR score when analysing a result.

#### Visualizing the quantization

From a quantization quality object you can also generate useful histograms of the distribution of the
float and the int data at the output of the model on a per-tensor or per-channel way. It can be a more
finer grain analysis of the quantization result than just the main scoring metrics.

```python
import mera
import numpy as np

# First, generate a quantized model
model_qtz = mera.ModelQuantizer("model_fp32.onnx").calibrate(...).quantize()

# Now get a list of images for evaluation. This should be a small list (1-5) of real images
# that have not been used on the calibration dataset. As before, real use case data should be used,
# using random data as a showcase.
eval_data = [np.random.randn(1, 224, 224, 3).astype(np.float32)]

# Compute the quality object, will run evaluation data on float32 and quantized model.
quality = model_qtz.measure_quantization_quality(eval_data)

# Some useful global summary numbers
print(f"Quantization PSNR: {quality.psnr}")
print(f"Quantization score: {quality.score}")
# Display all quality metrics
print(f"Full quantization statistics and metrics: {quality.to_dict()}")

# Generate histogram of output tensor
quality.plot_histogram("tensor_hist.png")
```

## Model compilation

With a MERA deployer object, you can call the `deploy()` method to deploy a specific model for a particular MERA target. The model needs to be imported with the `ModelLoader` class, as explained in the section above.

```python
import mera
from mera import Target, Platform

MODEL_PATH = "./model.pt"
INPUT_DESCRIPTION = {"input0": ((1, 224, 224, 3), "float32")}

# The MERA platform we are deploying for.
platform = Platform.DNAF200L0003
# Which of the MERA targets we want to deploy for.
target = Target.IP
# Architecture of the machine where inference will be run.
# Use "arm" for cross-compilation for aarch64 target environments.
host_arch = "x86"

DEPLOY_DIR = "./model_deployment"
with mera.TVMDeployer(DEPLOY_DIR) as deployer:
    # Or from_tflite() / from_quantized_mera()
    model = mera.ModelLoader(deployer).from_pytorch(MODEL_PATH, INPUT_DESCRIPTION)

    my_deploy = deployer.deploy(
        model,
        mera_platform = platform,
        target = target,
        host_arch = host_arch,
    )
```

## Inference

After deploying the model we can use the MERA deployment project to run inference with MERA.
For that you need a deploy object, which you can get it with the value returned from the `deploy()` method
or by loading a MERA project from disk. With that object you can create a model runner by calling
`get_runner()`

```python
import mera
from mera import Target, Platform

DEPLOY_DIR = "./model_deployment"
with mera.TVMDeployer(DEPLOY_DIR) as deployer:
    model = mera.ModelLoader(deployer).from_tflite("model.tflite")
    # 'my_deploy' is now a deploy object that can be used to perform inference.
    my_deploy = deployer.deploy(model, mera_platform=Platform.DNAF200L0003,
        target=Target.IP, host_arch="x86")
runner = my_deploy.get_runner()
```

If I have already performed the deployment (perhaps on a separate script or even machine),
I can load the deploy object directly by providing the path to the MERA deploy project where it was deployed using `mera.load_mera_deployment()`

```python
import mera
from mera import Target

DEPLOY_DIR = "./model_deployment"
# Only needed if my MERA deploy project contains more than 1 deployed target,
# in which case I specify which one I want to load.
target = Target.IP

my_deploy = mera.load_mera_deployment(DEPLOY_DIR, target=target)
runner = my_deploy.get_runner()
```

### Setting the inputs
With a model runner object, use the `set_input()` method.

```python
import mera
import numpy as np

my_deploy = mera.load_mera_deployment("./deploy_dir")
runner = my_deploy.get_runner()

# The input tensor used for inference
input_data = np.random.randn(1, 224, 224, 3)

runner.set_input(input_data)
```

When doing model fusion, the inputs of the fused model are the concatenation of the inputs of the models that are fused. We need to set each of the inputs.

```python
import mera
import numpy as np

DEPLOY_DIR = "./model_deployment"
with mera.TVMDeployer(DEPLOY_DIR) as deployer:
    model_a = mera.ModelLoader(deployer).from_tflite("./model_a.tflite")
    model_b = mera.ModelLoader(deployer).from_tflite("./model_b.tflite")
    fused_model = mera.ModelLoader(deployer).fuse_models([model_a, model_b], share_input=False)
    deploy_sim = deployer.deploy(fused_model, mera_platform=..., build_config=..., target=mera.Target.Simulator)
    runner = deploy_sim.get_runner()
    # set the inputs of the fused model
    input_names = list(fused_model.input_desc.keys())
    input_name_1 = input_names[0]
    input_name_2 = input_names[1]
    input_shape_1 = fused_model.get_input_shape(input_name_1)
    input_shape_2 = fused_model.get_input_shape(input_name_2)
    runner.set_input({input_name_1: np.random.randn(input_shape_1),
                      input_name_2: np.random.randn(input_shape_2)})
```

### Running a model
With a model runner object, use the `run()` method after setting input data.

```python
import mera
import numpy as np

my_deploy = mera.load_mera_deployment("./deploy_dir")
runner = my_deploy.get_runner()

input_data = np.random.randn(1, 224, 224, 3)
runner.set_input(input_data)

# Run inference
runner.run()
```

### Getting the outputs
With a model runner object, use the `get_outputs()` or `get_output()` method after running inference.
When doing model fusion, the outputs of the fused model are the concatenation of the outputs of the models that are fused.

```python
import mera
import numpy as np

my_deploy = mera.load_mera_deployment("./deploy_dir")
runner = my_deploy.get_runner()

input_data = np.random.randn(1, 224, 224, 3)
runner.set_input(input_data)

runner.run()

# Get all outputs as a list of NumPy tensors.
all_outputs = runner.get_outputs()
# Get only a single output, out_0 is the same as all_outputs[0]
out_0 = runner.get_output(0)
```
### Getting runtime latency metrics
You can get a breakdown of latency measurements on MERA IP by calling `get_runtime_metrics()` after running inference. Note that for a more complete list it is recommended to run with env `MERA_PROFILING=1`.

```python
import mera
import numpy as np

my_deploy = mera.load_mera_deployment("./deploy_dir")
input_data = np.random.randn(1, 224, 224, 3)

# Inference setup methods can also be chained together
runner = my_deploy.get_runner().set_input(input_data).run()

runtime_metrics = runner.get_runtime_metrics()
print(f"runtime_metrics = {runtime_metrics}")
```
