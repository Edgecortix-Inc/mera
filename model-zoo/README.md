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
| EdgeCortix&reg; MERA&trade; IP Simulator | DNAA600L0002 | x86               |

**Target**: generates deployments ready to be run on FPGA, ASIC or simulators. Other targets are allowed as, for example, the `IP` target which will deploy the model for physical FPGA/ASIC hardware or the `VerilatorSimulator` target that deploys a model for a cycle accurate ASIC simulation. For this example we will use the `Simulator` target which is a C++ based IP simulator.

**Platform**: specifies the hardware architecture. The current release of EdgeCortix&reg; MERA™ supports the following Dynamic Neural Accelerator architectures for FPGAs and ASICs:

| Platform Identifiers | Platform |      TOPS       |
| :------------------: | :------: | :-------------: |
|     DNAA800L0001     |   ASIC   | 78 <sub>1</sub> |
|     DNAA600L0002     |   ASIC   |       40        |
|     DNAA400L0001     |   ASIC   |      26.2       |
|     DNAF132S0001     |   FPGA   |       0.6       |
|     DNAF232S0002     |   FPGA   |       1.2       |
|     DNAF100L0003     |   FPGA   |       2.4       |
|     DNAF632L0003     |   FPGA   |       3.6       |
|     DNAF200L0003     |   FPGA   |       4.9       |

*Note<sub>1</sub> Recommended frequency for this platform is 1.2GHz*

**Host architecture**: specifies the architecture of the machine's CPU where the  EdgeCortix&reg; MERA™ IP is installed as a co-processor. For our current example, as the **Target** is `Simulator` , the simulation will run on CPU.

Now we are ready to launch the performance estimator script for all the models found in the specified directory. To specify the directory as an command line argument we use the flag `--modeldir` :

```bash
$ source performance_estimator.py --modeldir models/
...
Models compiled and executed. Results have been saved to the file: latencies.txt
```

Some remarks at this point. Please note that by default these models have been compiled and run for the platform **DNAA600L0002**. This platform architecture code it is by default assumed to run at **800MHz**. 

## Model list

| Model               | Framework   | Application                 | Batch size | Input resolution | Precision | Calibration data | Link                                                                  |
| ------------------- | ----------- | --------------------------- | :-------: | :--------------: | :-------: | :--------------: | --------------------------------------------------------------------- |
| ResNet18-v1.5       | Pytorch     | Classification              |     1     |     224x224      |   int8    |    Real data     | [link](https://edgecortix.box.com/s/f5tdfd6bgvhhvmcgm79vezk1ldbu77nc) |
| ResNet50-v1.5       | Pytorch     | Classification              |     1     |     224x224      |   int8    |    Real data     | [link](https://edgecortix.box.com/s/fhq3th1docshkdth66g5hlrqkb9z47x8) |
| YoloV3              | TFLite      | 2D Object detection         |     1     |     416x416      |   int8    |    Real data     | [link](https://edgecortix.box.com/s/bzpqnmkocme40grb5tfvrrdv2eaz9r4u) |
| Yolov5s             | TFLite      | 2D Object detection         |     1     |     448x448      |   int8    |    Real data     | [link](https://edgecortix.box.com/s/fgqwykxpw4xdy3tfc1mx4c82pk1ko8i9) |
| YoloV5m             | TFLite      | 2D Object detection         |     1     |     640x640      |   int8    |    Real data     | [link](https://edgecortix.box.com/s/3za6yi4g8jl263uxzix3awj73szq539g) |
| SFA3D               | PyTorch     | 3D LiDAR Object detection   |     1     |     608x608      |   int8    |    Real data     | [link](https://edgecortix.box.com/s/4smmrry1sgaj7imr548xmu4265fvwaua) |
| EfficientNet Lite 0 | TFLite      | Classification              |     1     |     240x240      |   int8    |    Real data     | [link](https://edgecortix.box.com/s/o7jvspeak5jqhysrcs58v6stgq4hfe3a) |
| EfficientNet Lite 2 | TFLite      | Classification              |     1     |     260x260      |   int8    |    Real data     | [link](https://edgecortix.box.com/s/ita019roakdur9gp4tqntxlrzc1qmglr) |
| EfficientNet Lite 3 | TFLite      | Classification              |     1     |     280x280      |   int8    |    Real data     | [link](https://edgecortix.box.com/s/jheozwqcalm2e10rdtlr4q4dp4fgpx1f) |
| EfficientNet Lite 4 | TFLite      | Classification              |     1     |     300x300      |   int8    |    Real data     | [link](https://edgecortix.box.com/s/m0k7vj8fwgykinmpt6enjxb6p0fppfvf) |
| EfficientNetV2 b0   | TFLite      | Classification              |     1     |     224x224      |   int8    |   Random data    | [link](https://edgecortix.box.com/s/9cgafa2w2ph0erqszhg0ck1qr9t9ap3x) |
| EfficientNetV2 b1   | TFLite      | Classification              |     1     |     224x224      |   int8    |   Random data    | [link](https://edgecortix.box.com/s/vjkorhuh6ihr0ulc1hjjcw0nyahvj72q) |
| EfficientNetV2 b2   | TFLite      | Classification              |     1     |     224x224      |   int8    |   Random data    | [link](https://edgecortix.box.com/s/75hgakm13i3kdezau3nqjke6mal46atp) |
| EfficientNetV2 b3   | TFLite      | Classification              |     1     |     224x224      |   int8    |   Random data    | [link](https://edgecortix.box.com/s/av9kgmnp4n0pl9tmi1qd0z2743ujc19i) |
| EfficientNetV2 s    | TFLite      | Classification              |     1     |     224x224      |   int8    |   Random data    | [link](https://edgecortix.box.com/s/ox54kf0pta2g1f39q7452so4fc35syy4) |
