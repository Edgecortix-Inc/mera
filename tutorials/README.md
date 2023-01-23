## Preconfigured MERA Tutorials
Here we provide a list of example code as a getting started guide on using the MERA&trade; compiler to deploy and run inference on typical deep neural network models using both PyTorch and TFLite frameworks. Check the corresponding docs for information about the tutorial contents.
### Tutorial List

- **PyTorch Resnet50 on Simulator** (`pytorch/resnet50_simulator.py`):

Contains an example on how to deploy and run a traced `resnet50` model in x86 host simulation.
Can be executed with the following command:

```
cd tutorials/pytorch
python3 resnet50_simulator.py
```

- **PyTorch Resnet50 on IP** (`pytorch/resnet50_ip.py`):

Contains an example on how to deploy and run a traced `resnet50` model in FPGA environment.
Needs to have FPGA runtime setup before running.
Can be executed with the following command:

```
cd tutorials/pytorch
# Needs to enable RUN_IP env in order to actually run the tutorial in HW
RUN_IP=1 python3 resnet50_ip.py
```

- **TFLite EfficientNet on Simulator** (`tflite/efficientnet_simulator.py`):

Contains an example on how to deploy and run a quantized `efficientnet-lite1` and `efficientnet-lite4` model in x86 host simulation and run an example object classification.
Can be executed with the following command:

```
cd tutorials/tflite
python3 efficientnet_simulator.py
```

- **TFLite EfficientNet on IP** (`tflite/efficientnet_ip.py`):

Contains an example on how to deploy and run a quantized `efficientnet-lite1` and `efficientnet-lite4` model in
FPGA environment and run an example object classification. Needs to have FPGA runtime setup before running.
Can be executed with the following command:

```
cd tutorials/tflite
# Needs to enable RUN_IP env in order to actually run the tutorial in HW
RUN_IP=1 python3 efficientnet_ip.py
```

- **Fused PyTorch Resnet18 + MobilenetV2 on Simulator** (`multi_models/fused_resnet_mobilenet_simulator.py`):

Contains an example on how to fuse two quantized PyTorch models (i.e., `resnet18` and `mobilenet_v2`) and then deploy the fused model in x86 host simulation.
Can be executed with the following command:

```
cd tutorials/multi_models
python3 fused_resnet_mobilenet_simulator.py
```
