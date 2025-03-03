import os
from pathlib import Path
import json
from comet_ml import Experiment
import boto3
import tarfile


THRESHOLD = 0.3


def evaluation_and_logging(pc_base_dir, exp):
    # read inputs
    predictions_dir = pc_base_dir / "input" / "predictions"
    predictions = []
    for file in predictions_dir.glob("*.out"):
        with open(file, "r") as f:
            predictions.append(json.loads(f.read()))
    ground_truth_dir = pc_base_dir / "input" / "ground_truth"
    with open(ground_truth_dir / "annotations.json") as f:
        ground_truth = json.load(f)

    # get the mapping
    s3 = boto3.client("s3")
    model_dir = pc_base_dir / "input" / "model"
    with tarfile.open(model_dir / "model.tar.gz", "r:gz") as tar:
        tar.extractall(model_dir)
    with open(os.path.join(model_dir, "labels_info.json"), "r") as f:
        labels_info = json.load(f)

    # evaluate
    true_positive = 0
    for i, p in enumerate(predictions):
        normalized_boxes, classes, scores = (
            p["normalized_boxes"],
            p["classes"],
            p["scores"],
        )
        high_confidence_mask = [score >= THRESHOLD for score in scores]
        normalized_boxes = [
            box for box, keep in zip(normalized_boxes, high_confidence_mask) if keep
        ]
        classes = [cls for cls, keep in zip(classes, high_confidence_mask) if keep]
        labels = [labels_info["labels"][int(cls)] for cls in classes]
        scores = [score for score, keep in zip(scores, high_confidence_mask) if keep]

        # TODO check if image filenames match.

        max_score_idx = scores.index(max(scores)) if scores else -1
        if max_score_idx == -1:
            continue
        predicted_label = int(labels[max_score_idx])
        gt_label = int(ground_truth["annotations"][i]["category_id"])
        if predicted_label == gt_label:
            true_positive += 1
    hit_rate = true_positive / len(predictions)

    # log
    exp.log_metric("hit_rate", hit_rate)
    exp.log_model(
        "tf_od_finetuned_model", str(model_dir / "model.tar.gz")
    )
    exp.log_parameter("training_job_name", os.environ["TRAINING_JOB_NAME"])


if __name__ == "__main__":
    exp = Experiment(
        api_key=os.environ["COMET_API_KEY"],
        project_name=os.environ["COMET_PROJECT_NAME"],
    )
    pc_base_dir = Path(os.environ["PC_BASE_DIR"])
    evaluation_and_logging(pc_base_dir, exp)
