from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.estimator import Estimator
from dotenv import load_dotenv
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.inputs import TrainingInput
import os
from sagemaker.workflow.steps import CacheConfig
from sagemaker.processing import ScriptProcessor
from sagemaker import image_uris, model_uris, script_uris, hyperparameters
import boto3
import tempfile
import tarfile
import json
import subprocess
from pathlib import Path
import pandas as pd


def run_pipeline():
    load_dotenv()

    cache_config = CacheConfig(enable_caching=True, expire_after="10d")

    sagemaker_session = PipelineSession(
        default_bucket=os.environ["S3_BUCKET_NAME"],
    )

    # create the classes mapping
    classes_id = list(map(int, os.environ["SAMPLE_CLASSES"].split(",")))
    s3 = boto3.client("s3")
    tmp_dir = Path(tempfile.mkdtemp())
    classes_dir = tmp_dir / "classes.txt"
    s3.download_file(
        os.environ["S3_BUCKET_NAME"],
        "dataset/classes.txt",
        classes_dir,
    )
    classes_df = pd.read_csv(classes_dir, sep=" ", names=["id", "class"], header=None)
    classes_mapping = {}
    for i, class_id in enumerate(classes_id):
        class_name = classes_df[classes_df["id"] == class_id]["class"].values[0]
        class_name = class_name.split(".")[-1]
        classes_mapping[class_id] = (i, class_name)
    mapping_dir = tmp_dir / "classes_mapping.json"
    with open(mapping_dir, "w") as f:
        json.dump(classes_mapping, f)
    s3.upload_file(
        mapping_dir, os.environ["S3_BUCKET_NAME"], "finetuning/classes_mapping.json"
    )

    # define the processor
    image_uri = "482497089777.dkr.ecr.eu-north-1.amazonaws.com/opencv:latest"
    processor = ScriptProcessor(
        image_uri=image_uri,
        instance_type=os.environ["PROCESSING_INSTANCE_TYPE"],
        instance_count=int(os.environ["PROCESSING_INSTANCE_COUNT"]),
        role=os.environ["SM_EXEC_ROLE"],
        sagemaker_session=sagemaker_session,
        env={
            "PC_BASE_DIR": os.environ["PC_BASE_DIR"],
            "SAMPLE_CLASSES": os.environ["SAMPLE_CLASSES"],
        },
        command=["python3"],
    )

    # defining the processing step
    processing_step = ProcessingStep(
        name="process-data",
        processor=processor,
        display_name="process data",
        description="This step is to prepare the dataset for finetuning.",
        inputs=[
            ProcessingInput(
                source=os.path.join(os.environ["S3_PROJECT_URI"], "dataset"),
                destination=os.path.join(os.environ["PC_BASE_DIR"], "dataset"),
                input_name="dataset",
                s3_data_type="S3Prefix",
                s3_input_mode="File",
            ),
            ProcessingInput(
                source=os.environ["S3_FINETUNING_CLASSES_MAPPING_URI"],
                destination=os.path.join(os.environ["PC_BASE_DIR"], "classes_mapping"),
                input_name="classes_mapping",
                s3_data_type="S3Prefix",
                s3_input_mode="File",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source=os.path.join(os.environ["PC_BASE_DIR"], "train"),
                destination=os.path.join(
                    os.environ["S3_PROJECT_URI"], "processing-step/train"
                ),
                output_name="train",
            ),
            ProcessingOutput(
                source=os.path.join(os.environ["PC_BASE_DIR"], "validation"),
                destination=os.path.join(
                    os.environ["S3_PROJECT_URI"], "processing-step/validation"
                ),
                output_name="validation",
            ),
            ProcessingOutput(
                source=os.path.join(os.environ["PC_BASE_DIR"], "model_input_feed"),
                destination=os.path.join(
                    os.environ["S3_PROJECT_URI"], "processing-step/model_input_feed"
                ),
                output_name="model_input_feed",
            ),
        ],
        code="src/processing.py",
        cache_config=cache_config,
    )

    image_uri = image_uris.retrieve(
        region=os.environ["AWS_REGION"],
        framework=None,
        model_id=os.environ["MODEL_ID"],
        model_version=os.environ["MODEL_VERSION"],
        image_scope="training",
        instance_type=os.environ["TRAINING_INSTANCE_TYPE"],
    )
    source_uri = script_uris.retrieve(
        model_id=os.environ["MODEL_ID"],
        model_version=os.environ["MODEL_VERSION"],
        script_scope="training",
    )
    model_uri = model_uris.retrieve(
        model_id=os.environ["MODEL_ID"],
        model_version=os.environ["MODEL_VERSION"],
        model_scope="training",
    )
    # modify the model artifacts to include the new classes
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        subprocess.run(
            [
                "aws",
                "s3",
                "cp",
                model_uri,
                f"{temp_dir}/",
            ],
            check=True,
        )
        zipped_model_fn = next(p for p in os.listdir(temp_dir) if p.endswith(".tar.gz"))
        model_path = os.path.join(temp_dir, zipped_model_fn)
        with tarfile.open(model_path, "r:gz") as tar:
            tar.extractall(temp_dir)
        # the labels json file
        labels_json_file = temp_dir / "labels_info.json"
        num_classes = 0
        with open(labels_json_file, "r+") as f:
            labels_dicttionary = json.load(f)
            labels_dicttionary["labels"] = (
                [v[-1] for _, v in classes_mapping.items()]
                + labels_dicttionary["labels"]
            )
            f.seek(0)
            json.dump(labels_dicttionary, f)
            f.truncate()
            num_classes = len(labels_dicttionary["labels"])
        # the pipeline config file
        config_file = temp_dir / "pipeline.config"
        import re

        with open(config_file, "r+") as f:
            config_content = f.read()
            config_content = re.sub(
                r"num_classes:\s*\d+", f"num_classes: {num_classes}", config_content
            )
            f.seek(0)
            f.write(config_content)
            f.truncate()
        modified_model_path = temp_dir / "model.tar.gz"
        with tarfile.open(modified_model_path, "w:gz") as tar:
            for item in os.listdir(temp_dir):
                if item.endswith(".tar.gz"):
                    continue
                item_path = temp_dir / item
                tar.add(item_path, arcname=os.path.relpath(item_path, temp_dir))
        # upload the modified model
        s3.upload_file(
            modified_model_path,
            os.environ["S3_BUCKET_NAME"],
            "finetuning/model/model.tar.gz",
        )
    hp = hyperparameters.retrieve_default(
        model_id=os.environ["MODEL_ID"], model_version=os.environ["MODEL_VERSION"]
    )

    from sagemaker.estimator import Estimator
    from sagemaker.utils import name_from_base

    training_job_name = name_from_base("tf-birds-detection-transfer-learning")

    training_metric_definitions = [
        {"Name": "val_localization_loss", "Regex": "Val_localization=([0-9\\.]+)"},
        {"Name": "val_classification_loss", "Regex": "Val_classification=([0-9\\.]+)"},
        {"Name": "train_loss", "Regex": "loss=([0-9\\.]+)."},
    ]

    estimator = Estimator(
        image_uri=image_uri,
        source_dir=source_uri,
        model_uri=os.environ["S3_FINETUNING_MODEL_URI"],
        entry_point="transfer_learning.py",
        role=os.environ["SM_EXEC_ROLE"],
        max_run=360000,
        instance_count=int(os.environ["TRAINING_INSTANCE_COUNT"]),
        instance_type=os.environ["TRAINING_INSTANCE_TYPE"],
        input_mode="File",
        base_job_name=training_job_name,
        sagemaker_session=sagemaker_session,
        output_path=os.path.join(os.environ["S3_PROJECT_URI"], "finetuning/output"),
        hyperparameters=hp,
        metric_definitions=training_metric_definitions,
    )

    training_step = TrainingStep(
        name="transfer-learning",
        step_args=estimator.fit(
            inputs={
                "train": TrainingInput(
                    s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[
                        "model_input_feed"
                    ].S3Output.S3Uri,
                    content_type="application/x-image",
                    s3_data_type="S3Prefix",
                ),
            },
        ),
        cache_config=cache_config,
    )

    # build the pipeline
    pipeline = Pipeline(
        name="tf-birds-detection-pipeline",
        steps=[
            processing_step,
            training_step,
        ],
        sagemaker_session=sagemaker_session,
    )

    pipeline.upsert(
        role_arn=os.environ["SM_EXEC_ROLE"],
        description="A pipeline to finetune the SSD Resnet50 model on the birds dataset.",
    )

    pipeline.start()


if __name__ == "__main__":
    run_pipeline()
