import os
import sys

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T

import mera
from mera import Target
from mera.mera_deployment import DeviceTarget

torch.set_grad_enabled(False)

# COCO classes
COCO_CLASSES = [
    "N/A",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# colors for visualization
COLORS = [
    [0.000, 0.447, 0.741],
    [0.850, 0.325, 0.098],
    [0.929, 0.694, 0.125],
    [0.494, 0.184, 0.556],
    [0.466, 0.674, 0.188],
    [0.301, 0.745, 0.933],
]


def flags_to_dict(flags):
    dt = {}
    for word in flags.split(","):
        k, v = word.split("=")
        dt[k.strip()] = v.strip()
    return dt


# standard PyTorch mean-std input image normalization
# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def plot_results(
    pil_img,
    prob,
    boxes,
    save_img_path,
    class_mapping=COCO_CLASSES,
):
    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), COLORS * 100):
        ax.add_patch(
            plt.Rectangle(
                (xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3
            )
        )
        cl = p.argmax()
        text = f"{class_mapping[cl]}: {p[cl]:0.2f}"
        ax.text(xmin, ymin, text, fontsize=15, bbox=dict(facecolor="yellow", alpha=0.5))
    plt.axis("off")
    plt.savefig(save_img_path)


def preprocessing(data):
    transform = T.Compose(
        [
            T.Resize((400, 600)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    prep_data = transform(data).unsqueeze(0)
    return prep_data


def postprocessing(pred, image_size=(224, 224), conf_thres=0.7):
    # keep only predictions with greater then confidence threshold
    outputs_0 = torch.from_numpy(pred[0])
    outputs_1 = torch.from_numpy(pred[1])

    probs = outputs_0.softmax(-1)[0, :, :-1]
    keep = probs.max(-1).values > conf_thres

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs_1[0, keep], image_size)

    return probs[keep], bboxes_scaled  # scores and boxes


def run_inference(image, predictor, model_flags):
    # --- take care of model flags
    model_flags = flags_to_dict(model_flags)
    conf_thres = float(model_flags["conf"])

    # --- preprocess
    preprocessed_input = preprocessing(image)

    # --- run raw inference
    runner = predictor.set_input(preprocessed_input).run()
    pred = runner.get_outputs()

    # --- postprocess
    scores, boxes = postprocessing(pred, image_size=image.size, conf_thres=conf_thres)

    return scores, boxes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        default="./data/input_image.jpg",
        type=str,
    )
    parser.add_argument(
        "--model_path",
        default="deploy_detr",
        type=str,
    )
    parser.add_argument(
        "--model_flags",
        default="a=0,conf=0.7",
        type=str,
    )
    parser.add_argument(
        "--save_path",
        default="result.png",
        type=str,
    )
    arg = parser.parse_args()

    image = Image.open(arg.input_path)
    print(f"Loaded input from {arg.input_path}...")

    target = Target.SimulatorBf16
    ip_deployment = mera.load_mera_deployment(arg.model_path, target=target)
    iprt = ip_deployment.get_runner()

    scores, boxes = run_inference(image, iprt, arg.model_flags)
    print(f"Got {len(scores)} bboxes")

    # Writes the output image into current working directory.
    plot_results(image, scores, boxes, arg.save_path)
    print(f"Saved result at {arg.save_path}")
