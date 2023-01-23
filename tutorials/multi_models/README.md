# Fusing Multiple Models for Compilation and Deployment

## Motivation
To fully utilize the compute resources of a large platform, it is desirable to fuse multiple models into a single model for compilation and deployment.

## User Interface
The entry for using this feature is `ModelLoader.fuse_models`, which takes in a list of models to be fused and an option of input sharing.
It returns a fused model.
The inputs of the fused model are the concatenation of the inputs of the models to be fused.
Similarly, the outputs of the fused model are the concatenation of the outputs of the models.
Input sharing is only supported when each model has exactly one input.
When input sharing is enabled, the fused model has one input.
Currently, it requires that the models to be fused should be from the same frontend, either Pytorch or Tflite.

## Example
The following code snippet shows how to fuse two models into a single model for compilation and deployment.

```python
import mera
import numpy as np

out_dir = "deployment_fused_resnet_mobilenet"
with mera.TVMDeployer(out_dir, overwrite=True) as deployer:
    effnet = mera.ModelLoader(deployer).from_tflite("efficientnetv2.tflite")
    yolo = mera.ModelLoader(deployer).from_tflite("yolov4.tflite")
    # effnet and yolo have different input shapes (224 x 224 vs. 416 x 416), so we cannot share the input
    fused_model = mera.ModelLoader(deployer).fuse_models([effnet, yolo], share_input=False)

    # deploy for Simulator target
    deploy_sim = deployer.deploy(fused_model, mera_platform=..., build_config=..., target=mera.Target.Simulator)
    # set the two inputs and run
    input_names = list(fused_model.input_desc.keys())
    input_name_1 = input_names[0]
    input_name_2 = input_names[1]
    input_shape_1 = fused_model.get_input_shape(input_name_1)
    input_shape_2 = fused_model.get_input_shape(input_name_2)
    sim_run = deploy_sim.get_runner().set_input({input_name_1: np.random(input_shape_1),
                                                 input_name_2: np.random(input_shape_2)}).run()
    # get the first output of effnet
    effnet_result_1 = sim_run.get_output(0)
    # get the first output of yolo
    yolo_result_1 = sim_run.get_output(1)
    # get the second output of yolo
    yolo_result_2 = sim_run.get_output(2)
```

For a complete and runnable example, please refer to `fused_resnet_mobilenet_simulator.py`.
