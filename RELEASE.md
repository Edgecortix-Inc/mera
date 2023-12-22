# Highlights for MERA 1.6.0

## Major features and improvements

* Added support for Ubuntu 22.04 and drop support for Ubuntu 18.04
* Improved support for BrainFloat16 precision for Sakura-II
* Added three Transformer models to the latency estimation demo
    * DE:TR, NanoGPT M and Mobile-ViT V1
* General improvements and bug-fixes
* Initial Performance Estimation support for SAKURA-II

# Highlights for MERA 1.5.0

## Major features and improvements

## Transformer model supported

* MERA has now support for transformer models. The model zoo has been updated with the following models:
  * nanoGPT language model
  * MobileViT v1 Vision Transformer for classification
  * DE:TR Vision Transformer for object detection

## BrainFloat16 DNA support

* Added half precision BrainFloat16 type support for compilation of float32 models and simulation.

## ONNX model loading support

* MERA can now load and deploy models with ONNX format.

# Highlights for MERA 1.3.0

## Major features and improvements

### Forward compatibility
* Official support for Ubuntu 20.04 LTS as well as Python 3.8. Compatibility with Ubuntu 18.04 LTS and Python 3.6 still provided.
* Newer versions of PyTorch and TensorFlow are now supported: PyTorch 1.12.1 and TensorFlow 2.9.0.

### MERA custom quantizer
* Featuring a built-in MERA custom quantizer premium (paid) feature. It allows users to feed MERA with FP32 models directly and apply quantization to it. This feature is disabled by default. MERA 1.3.0 still allows the deployment of pre-quantized models provided in the .mera format (see list in the Model Zoo).

### Multi-model support
* Support for multi-model deployments which allows the user to deploy more than one network at the same time. This can improve hardware utilization by running several models in parallel. Options to share the model inputs across multiple models is also available.

### Model Zoo update
* Model zoo update with new models and new .mera quantized sample models
  * SCI - Low light enhancement https://github.com/vis-opt-group/SCI
  * YoloV7 - https://github.com/WongKinYiu/yolov7
  * TinyYoloV7 - https://github.com/WongKinYiu/yolov7
  * YoloV4 - https://github.com/hunglc007/tensorflow-yolov4-tflite

### Profiling modifications
* Faster inference speeds on InterpreterHw and Interpreter target. Added feature for limiting the number of batches compiled at once for a MERA deployment. This is useful in cases where a model has very high batch dimension which could lead to the dramatic increase in number of instructions without corresponding gain in utilization.
* Improved inference latency in API for Xilinx devices. Now MERA interacts directly with a layer below of the OpenCL API. By reducing the number of layers between hardware and software the MERA Runtime improves latency.

# Highlights for MERA 1.2.0

## Major features and improvements

* Better exception handling
* Improved the graph partitioning algorithms
* Faster and better instruction scheduling
* Improved operator fusion algorithms
* Extended the MERA Model Zoo with new models
  * Super resolution
  * Semantic segmentation
  * Pose estimation
  * Low light enhancement
