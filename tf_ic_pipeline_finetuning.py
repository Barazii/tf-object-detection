from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep
from dotenv import load_dotenv
from sagemaker.processing import ProcessingInput, ProcessingOutput
import os
from sagemaker.workflow.steps import CacheConfig
from sagemaker.processing import ScriptProcessor
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.model import Model
from sagemaker.workflow.model_step import ModelStep
from sagemaker import image_uris, model_uris, script_uris, hyperparameters
from pathlib import Path
import tempfile
import subprocess
import boto3
from sagemaker.estimator import Estimator
from sagemaker.workflow.steps import TrainingStep
from sagemaker.inputs import TrainingInput


def run_pipeline():
    load_dotenv(override=True)

    cache_config = CacheConfig(enable_caching=False, expire_after="10d")

    sagemaker_session = PipelineSession(
        default_bucket=os.environ["S3_BUCKET_NAME"],
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
                source=os.path.join(os.environ["PC_BASE_DIR"], "finetuning"),
                destination=os.path.join(
                    os.environ["S3_PROJECT_URI"], "finetuning/input"
                ),
                output_name="finetuning",
            ),
        ],
        code="src/processing.py",
        cache_config=cache_config,
    )

    # training and finetuning
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
    hp = hyperparameters.retrieve_default(
        model_id=os.environ["MODEL_ID"], model_version=os.environ["MODEL_VERSION"]
    )
    hp["batch_size"] = 8
    hp["epochs"] = 7

    training_metric_definitions = [
        {"Name": "train:accuracy", "Regex": "accuracy: ([0-9\\.]+)"},
        {"Name": "train:loss", "Regex": "loss: ([0-9\\.]+)"},
        {"Name": "validation:accuracy", "Regex": "val_accuracy: ([0-9\\.]+)"},
        {"Name": "validation:loss", "Regex": "val_loss: ([0-9\\.]+)"},
    ]

    estimator = Estimator(
        image_uri=image_uri,
        source_dir=source_uri,
        model_uri=model_uri,
        entry_point="transfer_learning.py",
        role=os.environ["SM_EXEC_ROLE"],
        max_run=360000,
        instance_count=int(os.environ["TRAINING_INSTANCE_COUNT"]),
        instance_type=os.environ["TRAINING_INSTANCE_TYPE"],
        input_mode="File",
        base_job_name="no-hpo",
        sagemaker_session=sagemaker_session,
        hyperparameters=hp,
        metric_definitions=training_metric_definitions,
    )

    training_step = TrainingStep(
        name="transfer-learning",
        step_args=estimator.fit(
            inputs={
                "training": TrainingInput(
                    s3_data=os.path.join(os.environ["S3_PROJECT_URI"], "finetuning/input_data"),
                    content_type="application/x-image",
                    s3_data_type="S3Prefix",
                ),
            },
        ),
        cache_config=cache_config,
    )

    # create a sagemaker model instance out of the finetuned model to be deployed for inference
    image_uri = image_uris.retrieve(
        region=os.environ["AWS_REGION"],
        framework=None,
        image_scope="inference",
        model_id=os.environ["MODEL_ID"],
        model_version=os.environ["MODEL_VERSION"],
        instance_type=os.environ["INFERENCE_INSTANCE_TYPE"],
    )
    sourcedir_uri = script_uris.retrieve(
        model_id=os.environ["MODEL_ID"],
        model_version=os.environ["MODEL_VERSION"],
        script_scope="inference",
    )
    # download sourcedir, upalod to s3 (otherwise error for some reason)
    tmp_dir = Path(tempfile.mkdtemp())
    sourcedir = tmp_dir / "sourcedir"
    sourcedir.mkdir(exist_ok=True)
    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            sourcedir_uri,
            f"{sourcedir}/",
        ],
        check=True,
    )
    s3 = boto3.client("s3")
    s3.upload_file(
        f"{sourcedir}/sourcedir.tar.gz",
        os.environ["S3_BUCKET_NAME"],
        "finetuning/sourcedir/sourcedir.tar.gz",
    )

    model = Model(
        image_uri=image_uri,
        source_dir=os.path.join(
            os.environ["S3_PROJECT_URI"], "finetuning/sourcedir/sourcedir.tar.gz"
        ),
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        entry_point="inference.py",
        role=os.environ["SM_EXEC_ROLE"],
        name="tf-ic-birds-finetuned",
        sagemaker_session=sagemaker_session,
    )

    model_step = ModelStep(
        name="create-model",
        step_args=model.create(instance_type=os.environ["PROCESSING_INSTANCE_TYPE"]),
        depends_on=[training_step],
    )

    # build the pipeline
    pipeline = Pipeline(
        name="tf-birds-detection-pipeline",
        steps=[training_step, model_step],
        sagemaker_session=sagemaker_session,
    )

    pipeline.upsert(
        role_arn=os.environ["SM_EXEC_ROLE"],
        description="A pipeline to finetune a tf ic model on the birds dataset.",
    )

    pipeline.start()


if __name__ == "__main__":
    run_pipeline()
