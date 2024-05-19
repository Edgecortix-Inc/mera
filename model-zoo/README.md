# EdgeCortix&reg; MERA&trade; Model Zoo

This directory contains a list of deep learning models for different applications as classification, 2D object detection, 3D object detection, segmentation, super resolution, image enhancement and many more.

The models are self contained files already calibrated and quantized and provided in the standard file formats of the corresponding machine learning framework as for example TFLite in the case of TensorFlow or TorchScript in the case of PyTorch.

## Getting started

After installing EdgeCortix&reg; MERA&trade; following the [instructions](https://github.com/Edgecortix-Inc/mera/blob/main/README.md#installation-guide) in the main README file of this repository we are ready to start compiling and running models. For easy deployment, a script to run all the models found in a repository is provided.

### Models download

Models are separately stored separately from this repository. We should download the model files first, to do that, please run the provided download script:

```
$ python downloader.py
```

This script will download all the models into a newly created directory named `models/` under the same directory where the script ran.

### Performance estimator

Next we can start compiling and running models. The second script provided is named `performance_estimator.py` which will detect all the compatible files under the specified directory, later compiling and running them in the EdgeCortix&reg; MERA&trade; IP Simulator. More specifically this script will simulate models using:

| Target                                   | Platform     | Host architecture |
| ---------------------------------------- | ------------ | ----------------- |
| EdgeCortix&reg; MERA&trade; IP Simulator | SAKURA_1     | x86               |

**Target**: generates deployments ready to be run on FPGA, ASIC or simulators. Other targets are allowed as, for example, the `IP` target which will deploy the model for physical FPGA/ASIC hardware or the `VerilatorSimulator` target that deploys a model for a cycle accurate ASIC simulation. For this example we will use the `Simulator` target which is a C++ based IP simulator.

**Platform**: specifies the hardware architecture. The current release of EdgeCortix&reg; MERA™ supports the following Dynamic Neural Accelerator architectures for ASICs:

| Platform Identifiers  | Platform | TOPS               | TFLOPS |
|:---------------------:|:--------:|:------------------:|:------:|
|          SAKURA_1     |  ASIC    |  40                | N/A    |
|         SAKURA_II     |  ASIC    |  60                | 30     |


**Host architecture**: specifies the architecture of the machine's CPU where the  EdgeCortix&reg; MERA™ IP is installed as a co-processor. For our current example, as the **Target** is `Simulator` , the simulation will run on CPU.

Now we are ready to launch the performance estimator script for all the models found in the specified directory. To specify the directory as an command line argument we use the flag `--modeldir` :

```bash
$ python performance_estimator.py --models models/
...
Models compiled and executed. Results have been saved to the file: latencies.txt
```

Some remarks at this point. Please note that by default these models have been compiled and run for the platform **DNAA600L0002**. This platform architecture code it is by default assumed to run at **800MHz**. 

## Model list

 | Id       | Model               | Framework   | Application                       | Batch size | Input resolution | Precision | Calibration data | Link                                                                  |
 | -------  | ------------------- | ----------- | --------------------------------- | :-------: | :--------------: | :-------: | :--------------: | --------------------------------------------------------------------- |
 |    1     | ResNet18-v1.5       | Pytorch     | Classification                    |     1     |     224x224      |   int8    |    Real data     | [link](https://edgecortix.box.com/s/f5tdfd6bgvhhvmcgm79vezk1ldbu77nc) |
 |    2     | ResNet50-v1.5       | Pytorch     | Classification                    |     1     |     224x224      |   int8    |    Real data     | [link](https://edgecortix.box.com/s/fhq3th1docshkdth66g5hlrqkb9z47x8) |
 |    3     | YoloV3              | TFLite      | 2D Object detection               |     1     |     416x416      |   int8    |    Real data     | [link](https://edgecortix.box.com/s/bzpqnmkocme40grb5tfvrrdv2eaz9r4u) |
 |    4     | Yolov5s             | TFLite      | 2D Object detection               |     1     |     448x448      |   int8    |    Real data     | [link](https://edgecortix.box.com/s/fgqwykxpw4xdy3tfc1mx4c82pk1ko8i9) |
 |    5     | YoloV5m             | TFLite      | 2D Object detection               |     1     |     640x640      |   int8    |    Real data     | [link](https://edgecortix.box.com/s/3za6yi4g8jl263uxzix3awj73szq539g) |
 |    6     | SFA3D               | PyTorch     | 3D LiDAR Object detection         |     1     |     608x608      |   int8    |    Real data     | [link](https://edgecortix.box.com/s/4smmrry1sgaj7imr548xmu4265fvwaua) |
 |    7     | EfficientNet Lite 0 | TFLite      | Classification                    |     1     |     240x240      |   int8    |    Real data     | [link](https://edgecortix.box.com/s/o7jvspeak5jqhysrcs58v6stgq4hfe3a) |
 |    8     | EfficientNet Lite 2 | TFLite      | Classification                    |     1     |     260x260      |   int8    |    Real data     | [link](https://edgecortix.box.com/s/ita019roakdur9gp4tqntxlrzc1qmglr) |
 |    9     | EfficientNet Lite 3 | TFLite      | Classification                    |     1     |     280x280      |   int8    |    Real data     | [link](https://edgecortix.box.com/s/jheozwqcalm2e10rdtlr4q4dp4fgpx1f) |
 |   10     | EfficientNet Lite 4 | TFLite      | Classification                    |     1     |     300x300      |   int8    |    Real data     | [link](https://edgecortix.box.com/s/m0k7vj8fwgykinmpt6enjxb6p0fppfvf) |
 |   11     | EfficientNetV2 b0   | TFLite      | Classification                    |     1     |     224x224      |   int8    |   Random data    | [link](https://edgecortix.box.com/s/9cgafa2w2ph0erqszhg0ck1qr9t9ap3x) |
 |   12     | EfficientNetV2 b1   | TFLite      | Classification                    |     1     |     224x224      |   int8    |   Random data    | [link](https://edgecortix.box.com/s/vjkorhuh6ihr0ulc1hjjcw0nyahvj72q) |
 |   13     | EfficientNetV2 b2   | TFLite      | Classification                    |     1     |     224x224      |   int8    |   Random data    | [link](https://edgecortix.box.com/s/75hgakm13i3kdezau3nqjke6mal46atp) |
 |   14     | EfficientNetV2 b3   | TFLite      | Classification                    |     1     |     224x224      |   int8    |   Random data    | [link](https://edgecortix.box.com/s/av9kgmnp4n0pl9tmi1qd0z2743ujc19i) |
 |   15     | EfficientNetV2 s    | TFLite      | Classification                    |     1     |     224x224      |   int8    |   Random data    | [link](https://edgecortix.box.com/s/ox54kf0pta2g1f39q7452so4fc35syy4) |
 |   16     | MonoDepth           | PyTorch     | Monocular depth estimation        |     1     |     384x288      |   int8    |   Real data      | [link](https://edgecortix.box.com/s/uv0vvqkyp3rx7v1lgapv6yufrcurastd) |
 |   17     | U-Net               | TFLite      | Semantic segmentation             |     1     |     128x128      |   int8    |   Real data      | [link](https://edgecortix.box.com/s/2y4d405sqpmrhsvg0mrefznj7tj1nrve) |
 |   18     | MoveNet Thunder     | TFLite      | Pose estimation                   |     1     |     256x256      |   int8    |   Real data      | [link](https://edgecortix.box.com/s/kh0mghv2yd88mstucql1tsq6t0jn2w18) |
 |   19     | YoloV4 Tiny         | TFLite      | 2D Object detection               |     1     |     640x640      |   int8    |   Real data      | [link](https://edgecortix.box.com/s/1wokg6m238hesq0w4uat07nhewasmn21) |
 |   20     | DeepLabEdgeTPU m    | TFLite      | Semantic segmentation             |     1     |     512x512      |   int8    |   Real data      | [link](https://edgecortix.box.com/s/28dcbrtby7whzpvcmoldgqyfjs2d4slj) |
 |   21     | DeepLabEdgeTPU s    | TFLite      | Semantic segmentation             |     1     |     512x512      |   int8    |   Real data      | [link](https://edgecortix.box.com/s/57jrlqgsxy68w3i5gvh2m0qn7lay8f3o) |
 |   22     | MoveNet Lighting    | TFLite      | Pose estimation                   |     1     |     192x192      |   int8    |   Real data      | [link](https://edgecortix.box.com/s/oj2g7rwpk3n0t2fphfx65p96i4l4ip7e) |
 |   23     | MobileNetV2 SSD     | PyTorch     | 2D Object detection               |     1     |     640x480      |   int8    |   Real data      | [link](https://edgecortix.box.com/s/jcifbv6tkrcinqczoalsemel4nm9fk6w) |
 |   24     | DeepLabEdgeTPU xs   | TFLite      | Semantic segmentation             |     1     |     512x512      |   int8    |   Real data      | [link](https://edgecortix.box.com/s/150wnhkxvdaja9fbr93v76x1jomtrhs4) |
 |   25     | GladNet             | TFLite      | Low light enhancement             |     1     |     640x480      |   int8    |   Real data      | [link](https://edgecortix.box.com/s/hg6zh4bu8a1cp701zc16ro410yi69lkl) |
 |   26     | ABPN                | TFLite      | Super resolution                  |     1     |     640x360      |   int8    |   Real data      | [link](https://edgecortix.box.com/s/x9hxsd5030u3slbnj847q9kgudk6bx6m) |
 |   27     | YoloV7              | MERA        | 2D Object detection               |     1     |     640x640      |   int8    |   Real data      | [link](https://edgecortix.box.com/s/13fx4hc0pkmokhppa6ats6eya15ad8a9) |
 |   28     | YoloV4              | TFLite      | 2D Object detection               |     1     |     416x416      |   int8    |   Real data      | [link](https://edgecortix.box.com/s/bnqxgw7vm5tu2651z8mleszodlbdh3xi) |
 |   29     | SCI                 | MERA        | Low light enhancement             |     1     |     1280x720     |   int8    |   Real data      | [link](https://edgecortix.box.com/s/pv92y5rf33en6qrm2zb727l5kc7ip598) |
 |   30     | NanoGPT M           | ONNX        | Language model transformer        |     1     |     64           |   FP32    |   Real data      | [link](https://edgecortix.box.com/s/hko9ps3064bq9svzrpx0uo0w7p8aa5xf) |
 |   31     | DE:TR               | ONNX        | Vision Transformer detection      |     1     |     600x400      |   FP32    |   Real data      | [link](https://edgecortix.box.com/s/iw5250rfi3pk3wyzicpt6br9glj38ouh) |
 |   32     | Mobile ViT V1       | ONNX        | Vision Transformer classification |     1     |     256x256      |   FP32    |   Real data      | [link](https://edgecortix.box.com/s/no3uxexvarzzsh5wqbbn095hfsirp2m1) |
