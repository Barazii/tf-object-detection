import sagemaker
import dotenv
from sagemaker.predictor import Predictor
from sagemaker.serializers import IdentitySerializer
from sagemaker.deserializers import JSONDeserializer
import os
import tempfile
import tarfile
import json
from pathlib import Path
import time
import boto3
import pandas as pd


def query(model_predictor, image_file_name):
    with open(image_file_name, "rb") as file:
        input_img_rb = file.read()

    query_response = model_predictor.predict(
        input_img_rb,
        {
            "ContentType": "application/x-image",
            # "Accept": "application/json;verbose;n_predictions=5",
        },
    )
    return query_response


def parse_response(query_response, classes_mapping, labels_info):
    normalized_boxes, classes, scores = (
        query_response["normalized_boxes"],
        query_response["classes"],
        query_response["scores"],
    )
    # take predictions with scores >= 0.5 only
    high_confidence_mask = [score >= 0.3 for score in scores]
    normalized_boxes = [
        box for box, keep in zip(normalized_boxes, high_confidence_mask) if keep
    ]
    classes = [cls for cls, keep in zip(classes, high_confidence_mask) if keep]
    scores = [score for score, keep in zip(scores, high_confidence_mask) if keep]
    # map
    # labels = [classes_mapping[labels_info["labels"][int(cls)]] for cls in classes]
    labels = [labels_info["labels"][int(cls)] for cls in classes]

    return normalized_boxes, labels, scores


import matplotlib.patches as patches
from matplotlib import pyplot as plt
from PIL import Image
from PIL import ImageColor
import numpy as np
import time


def display_predictions(
    img_jpg, normalized_boxes, classes_names, confidences, output_path
):
    colors = list(ImageColor.colormap.values())
    image_np = np.array(Image.open(img_jpg))
    plt.figure(figsize=(20, 20))
    ax = plt.axes()
    ax.imshow(image_np)

    for idx in range(len(normalized_boxes)):
        left, bot, right, top = normalized_boxes[idx]
        x, w = [val * image_np.shape[1] for val in [left, right - left]]
        y, h = [val * image_np.shape[0] for val in [bot, top - bot]]
        color = colors[hash(classes_names[idx]) % len(colors)]
        rect = patches.Rectangle(
            (x, y), w, h, linewidth=3, edgecolor=color, facecolor="none"
        )
        ax.add_patch(rect)
        ax.text(
            x,
            y,
            "{} {:.0f}%".format(classes_names[idx], confidences[idx] * 100),
            bbox=dict(facecolor="white", alpha=0.5),
        )

    plt.savefig(f"{output_path}/predictions_{int(time.time())}.png")


def run_test():
    dotenv.load_dotenv(override=True)
    sagemaker_session = sagemaker.Session()

    predictor = Predictor(
        endpoint_name=os.environ["ENDPOINT_NAME"],
        sagemaker_session=sagemaker_session,
        serializer=IdentitySerializer("image/jpeg"),
        deserializer=JSONDeserializer(),
    )

    # labels info and classes mapping
    s3 = boto3.client("s3")
    temp_dir = tempfile.mkdtemp()
    model_dir = os.path.join(temp_dir, "model.tar.gz")
    endpoint_config_name = boto3.client("sagemaker").describe_endpoint(
        EndpointName=os.environ["ENDPOINT_NAME"]
    )["EndpointConfigName"]
    production_variant = boto3.client("sagemaker").describe_endpoint_config(
        EndpointConfigName=endpoint_config_name
    )["ProductionVariants"][0]
    model_name = production_variant["ModelName"]
    model_info = boto3.client("sagemaker").describe_model(ModelName=model_name)
    model_uri = model_info["PrimaryContainer"]["ModelDataUrl"]
    s3_url_parts = model_uri.replace("s3://", "").split("/")
    bucket_name = s3_url_parts[0]
    key = "/".join(s3_url_parts[1:])
    s3.download_file(bucket_name, key, model_dir)
    with tarfile.open(model_dir, "r:gz") as tar:
        tar.extractall(temp_dir)
    with open(os.path.join(temp_dir, "labels_info.json"), "r") as f:
        labels_info = json.load(f)
    classes_id = list(map(int, os.environ["SAMPLE_CLASSES"].split(",")))
    tmp_dir = Path(tempfile.mkdtemp())
    classes_dir = tmp_dir / "classes.txt"
    s3.download_file(
        os.environ["S3_BUCKET_NAME"],
        "dataset/classes.txt",
        classes_dir,
    )
    classes_df = pd.read_csv(classes_dir, sep=" ", names=["id", "class"], header=None)
    classes_mapping = {}
    for class_id in classes_id:
        class_name = classes_df[classes_df["id"] == class_id]["class"].values[0]
        class_name = class_name.split(".")[-1]
        classes_mapping[class_id] = class_name

    # test images
    input_images_path = "test_input_images"
    output_path = "test_output_images"
    os.makedirs(output_path, exist_ok=True)
    for image in os.listdir(input_images_path):
        if image.lower().endswith((".jpg", ".jpeg")):
            image_path = os.path.join(input_images_path, image)
            response = query(predictor, image_path)
            nbb, lbl, scr = parse_response(response, classes_mapping, labels_info)
            display_predictions(image_path, nbb, lbl, scr, output_path)


if __name__ == "__main__":
    run_test()
