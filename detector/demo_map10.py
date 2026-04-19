"""
Simple inference demo for the map_10 class setup.

Usage examples (run from detector/):
    python demo_map10.py --image ../data/batch_1/000001.jpg --model taco20260418T2028
    python demo_map10.py --image ../data/batch_1/000001.jpg --model ./models/logs/taco20260418T2028/mask_rcnn_taco_0032.weights.h5
"""

import argparse
import csv
import os
import re

import numpy as np
from PIL import Image

from config import Config
from model import MaskRCNN


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CLASS_MAP = os.path.join(SCRIPT_DIR, "taco_config", "map_10.csv")
DEFAULT_LOGS_DIR = os.path.join(SCRIPT_DIR, "models", "logs")


def load_target_class_names(class_map_path):
    mapped_names = []
    with open(class_map_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if len(row) >= 2 and row[1].strip():
                mapped_names.append(row[1].strip())

    if not mapped_names:
        raise ValueError("No classes found in class map: {}".format(class_map_path))

    target_names = sorted(set(mapped_names))
    if "Background" in target_names:
        target_names.remove("Background")

    # Dataset class_ids start from 1, with 0 reserved for background.
    return ["BG"] + target_names


def extract_checkpoint_epoch(checkpoint_path):
    filename = os.path.basename(checkpoint_path)
    match = re.search(r"_(\d{4})(?:\.weights)?\.h5$", filename)
    return int(match.group(1)) if match else None


def select_latest_checkpoint(checkpoint_dir):
    checkpoints = [
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("mask_rcnn") and f.endswith(".h5") and os.path.isfile(os.path.join(checkpoint_dir, f))
    ]
    if not checkpoints:
        return None

    epoch_checkpoints = []
    for filename in checkpoints:
        epoch = extract_checkpoint_epoch(filename)
        if epoch is not None:
            epoch_checkpoints.append((epoch, filename))

    if epoch_checkpoints:
        epoch_checkpoints = sorted(epoch_checkpoints, key=lambda item: item[0])
        return os.path.join(checkpoint_dir, epoch_checkpoints[-1][1])

    checkpoints = sorted(checkpoints)
    return os.path.join(checkpoint_dir, checkpoints[-1])


def resolve_model_path(model_arg, logs_dir):
    expanded_model_arg = os.path.abspath(os.path.expanduser(model_arg))
    if os.path.isfile(expanded_model_arg):
        return expanded_model_arg

    run_dir_candidates = [
        expanded_model_arg,
        os.path.join(logs_dir, model_arg),
    ]

    for run_dir in run_dir_candidates:
        run_dir = os.path.abspath(run_dir)
        if os.path.isdir(run_dir):
            checkpoint_path = select_latest_checkpoint(run_dir)
            if checkpoint_path is not None:
                return checkpoint_path
            raise FileNotFoundError(
                "No checkpoint files found in run directory: {}".format(run_dir)
            )

    raise FileNotFoundError(
        "Model not found. Provide either a .h5 file path or a run directory name under {}"
        .format(logs_dir)
    )


def load_image_rgb(image_path):
    image = Image.open(image_path)
    exif = image._getexif()
    if exif:
        exif = dict(exif.items())
        if 274 in exif:
            if exif[274] == 3:
                image = image.rotate(180, expand=True)
            if exif[274] == 6:
                image = image.rotate(270, expand=True)
            if exif[274] == 8:
                image = image.rotate(90, expand=True)

    image = image.convert("RGB")
    return np.array(image)


def class_name_from_id(class_names, class_id):
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    return "Unknown({})".format(class_id)


def main():
    parser = argparse.ArgumentParser(description="Run map_10 inference on one image and print predicted classes.")
    parser.add_argument("--image", required=True, help="Path to input image")
    parser.add_argument(
        "--model",
        required=True,
        help="Path to .h5 checkpoint, or model run folder name under models/logs",
    )
    parser.add_argument("--class_map", default=DEFAULT_CLASS_MAP, help="Path to class mapping CSV")
    parser.add_argument("--logs", default=DEFAULT_LOGS_DIR, help="Model logs directory")
    parser.add_argument("--top_k", type=int, default=5, help="Number of top detections to print")
    parser.add_argument("--min_conf", type=float, default=0.0, help="Minimum detection confidence")
    parser.add_argument(
        "--score_ratio",
        action="store_true",
        help="Use score ratio ranking (score/background) like project evaluation mode",
    )

    args = parser.parse_args()
    if args.top_k < 1:
        raise ValueError("--top_k must be >= 1")

    class_names = load_target_class_names(args.class_map)
    num_classes = len(class_names)

    class DemoMap10Config(Config):
        NAME = "taco_demo_map10"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = num_classes
        USE_OBJECT_ZOOM = False
        DETECTION_MIN_CONFIDENCE = max(0.0, float(args.min_conf))
        DETECTION_SCORE_RATIO = args.score_ratio

    config = DemoMap10Config()
    model = MaskRCNN(mode="inference", config=config, model_dir=args.logs)

    model_path = resolve_model_path(args.model, args.logs)
    model.load_weights(model_path, model_path, by_name=True)

    image = load_image_rgb(args.image)
    result = model.detect([image], verbose=0)[0]

    class_ids = result["class_ids"]
    scores = result["scores"]

    print("Image: {}".format(args.image))
    print("Checkpoint: {}".format(model_path))
    print("Classes: {}".format(", ".join(class_names[1:])))

    if class_ids.shape[0] == 0:
        print("No detections found.")
        return

    order = np.argsort(scores)[::-1]
    order = order[:min(args.top_k, order.shape[0])]

    best_idx = order[0]
    best_class_id = int(class_ids[best_idx])
    best_score = float(scores[best_idx])
    best_class_name = class_name_from_id(class_names, best_class_id)
    print("Predicted class: {} (class_id={}, score={:.4f})".format(
        best_class_name, best_class_id, best_score))

    print("Top detections:")
    for rank, idx in enumerate(order, start=1):
        class_id = int(class_ids[idx])
        class_name = class_name_from_id(class_names, class_id)
        score = float(scores[idx])
        print("{:>2}. {} (class_id={}, score={:.4f})".format(
            rank, class_name, class_id, score))


if __name__ == "__main__":
    main()
