
# DE:TR - Vision Transformer demo 

Demo of the vision transformer DE:TR model for object detection, from [facebookresearch/detr](https://github.com/facebookresearch/detr) repo.
This demo shows how to deploy the model and do inference using the EdgeCortix&reg; custom BrainFloat-16 precision arithmethic on the Simulator target.

## How to run the demo

First download the model from the EdgeCortix&reg; MERA&trade; Model Zoo with the following command:

```bash
python downloader.py
```

Then compile the model:

```bash
python deploy.py
```

Finally run the demo on Simulator:

```bash
python demo_model.py
```

