import subprocess
from typing import List
import os

import os
import json
import torch
import sys
from PIL import Image
from tqdm import tqdm

import supervision as sv
def run_shell_command(command: List[str], working_directory=None) -> None:
    subprocess.run(
        command, check=True, text=True, stdout=None, stderr=None, cwd=working_directory
    )

def download_file(url: str, output_filename: str) -> None:
    command = ["wget", url, "-O", output_filename]
    subprocess.run(
        command, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
from supervision.utils.file import read_json_file
from supervision.dataset.formats.coco import coco_categories_to_classes, build_coco_class_index_mapping
from mmdet.apis import inference_detector, init_detector

def get_coco_class_index_mapping(annotations_path: str):
    coco_data = read_json_file(annotations_path)
    classes = coco_categories_to_classes(coco_categories=coco_data["categories"])
    class_mapping = build_coco_class_index_mapping(
        coco_categories=coco_data["categories"], target_classes=classes
    )
    return class_mapping

def print_10f(stats):
    metrics = ["AP[0.50:0.95", "AP@.50", "AP@.75", "AP_small", "AP_medium", "AP_large"]
    for idx, metric_name in enumerate(metrics):
        print(f"{metric_name}: {stats[idx]:.10f}")

def run_on_image(model, image) -> sv.Detections:
    result = inference_detector(model, image)
    detections = sv.Detections.from_mmdetection(result)
    return detections
def download_weight(config_name):
    run_shell_command(
        [
            sys.executable,
            "-m",
            "mim",
            "download",
            "mmyolo",
            "--config",
            config_name,
            "--dest",
            "mmyolo-weights/",
        ]
    )
def main():
    #install coco api
    run_shell_command(["git", "clone", "https://github.com/cocodataset/cocoapi"])
    run_shell_command(["make"], working_directory="cocoapi/PythonAPI")
    run_shell_command(["python", "setup.py", "build_ext", "install"], working_directory="cocoapi/PythonAPI")
    run_shell_command(["cd", ".."], working_directory="cocoapi/PythonAPI")

    os.mkdir("coco_dataset")

    # Download images (val2017)
    run_shell_command(['wget', 'http://images.cocodataset.org/zips/val2017.zip'])
    run_shell_command(['unzip', 'val2017.zip', '-d', 'coco_dataset/'])
    run_shell_command(['rm', 'val2017.zip'])


    # Download annotations
    run_shell_command(['wget', 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'])
    run_shell_command(['unzip', 'annotations_trainval2017.zip', '-d', 'coco_dataset/'])
    run_shell_command(['rm', 'annotations_trainval2017.zip'])
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval


    MODEL_DICT = {
        "rtmdet_tiny_syncbn_fast_8xb32-300e_coco": {
            "model_name": "RTMDet-tiny",
            "config": "./mmyolo-weights/rtmdet_tiny_syncbn_fast_8xb32-300e_coco.py",
            "checkpoint_file": "./mmyolo-weights/rtmdet_tiny_syncbn_fast_8xb32-300e_coco_20230102_140117-dbb1dc83.pth",  # noqa: E501 // docs
        },
        "rtmdet_s_syncbn_fast_8xb32-300e_coco": {
            "model_name": "RTMDet-s",
            "config": "./mmyolo-weights/rtmdet_s_syncbn_fast_8xb32-300e_coco.py",
            "checkpoint_file": "./mmyolo-weights/rtmdet_s_syncbn_fast_8xb32-300e_coco_20221230_182329-0a8c901a.pth",  # noqa: E501 // docs
        },
        "rtmdet_m_syncbn_fast_8xb32-300e_coco": {
            "model_name": "RTMDet-m",
            "config": "./mmyolo-weights/rtmdet_m_syncbn_fast_8xb32-300e_coco.py",
            "checkpoint_file": "./mmyolo-weights/rtmdet_m_syncbn_fast_8xb32-300e_coco_20230102_135952-40af4fe8.pth",  # noqa: E501 // docs
        },
        "rtmdet_l_syncbn_fast_8xb32-300e_coco": {
            "model_name": "RTMDet-l",
            "config": "./mmyolo-weights/rtmdet_l_syncbn_fast_8xb32-300e_coco.py",
            "checkpoint_file": "./mmyolo-weights/rtmdet_l_syncbn_fast_8xb32-300e_coco_20230102_135928-ee3abdc4.pth",  # noqa: E501 // docs
        },
        "rtmdet_x_syncbn_fast_8xb32-300e_coco": {
            "model_name": "RTMDet-x",
            "config": "./mmyolo-weights/rtmdet_x_syncbn_fast_8xb32-300e_coco.py",
            "checkpoint_file": "./mmyolo-weights/rtmdet_x_syncbn_fast_8xb32-300e_coco_20221231_100345-b85cd476.pth",  # noqa: E501 // docs
        },
    }

    LICENSE = "GPL-3.0"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CONFIDENCE_THRESHOLD = 0.001

    RUN_PARAMETERS = dict(
        imgsz=640,
        conf=CONFIDENCE_THRESHOLD,
    )
    GIT_REPO_URL = "https://github.com/open-mmlab/mmyolo/tree/main/configs/rtmdet"
        # Initialize the COCO class from pycocotools
    annotation_file = "coco_dataset/annotations/instances_val2017.json"
    images_folder = "coco_dataset/val2017"
    coco_gt = COCO(annotation_file)
    class_mapping = get_coco_class_index_mapping(annotation_file)
    inv_class_mapping = {v: k for k, v in class_mapping.items()}

    # Get all image IDs
    image_ids = coco_gt.getImgIds()

    # Run evaluation for each model
    for model_id in MODEL_DICT:
        # Initialize the COCO dataset from the annotations file
        coco_gt = COCO(annotation_file)
        image_ids = coco_gt.getImgIds()

        # Load model
        model_values = MODEL_DICT[model_id]

        download_weight(model_id)

        model = init_detector(
            model_values["config"], model_values["checkpoint_file"], DEVICE
        )

        print(f"\nEvaluating model: {model_id}")

        predictions = []
        for img_id in tqdm(image_ids, total=len(image_ids)):
            # Get image path from img_id
            img_info = coco_gt.loadImgs(img_id)[0]
            file_name = img_info['file_name']
            img_path = os.path.join(images_folder, file_name)

            # Load image with PIL
            image = Image.open(img_path).convert("RGB")

            # Run inference
            detections = run_on_image(model, image)
            result = detections[detections.confidence > CONFIDENCE_THRESHOLD]
            class_id = result.class_id.astype(int)
            xyxy = result.xyxy
            xywh = xyxy.copy()
            xywh[:, 2:4] -= xywh[:, 0:2]
            confidences = result.confidence

            # Map predicted class ids to coco class ids
            category_ids = [inv_class_mapping[i] for i in class_id]

            # Create predictions in the format required by pycocotools
            for cat_id, bbox, score in zip(category_ids, xywh, confidences):
                detections = {
                    "image_id": img_id,
                    "category_id": cat_id,
                    "bbox": bbox.tolist(),
                    "score": score.item(),
                }
                predictions.append(detections)

        # COCO requires predictions in a json file
        coco_prediction_file = f"results_pycocotools_coco_predictions_{model_id}.json"
        with open(coco_prediction_file, "w") as f:
            json.dump(predictions, f, indent=None)

        # Load the predictions that we have just saved
        coco_det = coco_gt.loadRes(coco_prediction_file)
        # Create COCOeval object
        cocoEval = COCOeval(coco_gt, coco_det, iouType="bbox")
        # Evaluate and print results
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        # Print AP values only with 10f precision
        print_10f(cocoEval.stats)