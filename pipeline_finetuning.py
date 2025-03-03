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
import subprocess
from pathlib import Path


HPO = True


def run_pipeline():
    load_dotenv()

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

    from sagemaker.tuner import (
        ContinuousParameter,
        CategoricalParameter,
    )

    amt_metric_definitions = {
        "metrics": [
            {
                "Name": "validation:localization_loss",
                "Regex": "Val_localization=([0-9\\.]+)",
            }
        ],
        "type": "Minimize",
    }
    hyperparameter_ranges = {
        "learning_rate": ContinuousParameter(0.000001, 0.5, scaling_type="Logarithmic"),
        "batch_size": CategoricalParameter([8, 16, 32]),
        "optimizer": CategoricalParameter(
            ["sgd", "adam", "rmsprop", "nesterov", "adagrad", "adadelta"]
        ),
        "momentum": ContinuousParameter(0.0001, 0.9, scaling_type="Logarithmic"),
        "regularizers_l2": ContinuousParameter(
            0.00001, 0.9, scaling_type="Logarithmic"
        ),
        "beta_1": ContinuousParameter(0.0001, 0.9, scaling_type="Logarithmic"),
        "beta_2": ContinuousParameter(0.0001, 0.9, scaling_type="Logarithmic"),
    }
    max_jobs = 6
    max_parallel_jobs = 2

    from sagemaker.utils import name_from_base

    training_job_name = name_from_base("no-hpo")

    training_metric_definitions = [
        {"Name": "val_localization_loss", "Regex": "Val_localization=([0-9\\.]+)"},
        {"Name": "val_classification_loss", "Regex": "Val_classification=([0-9\\.]+)"},
        {"Name": "train_loss", "Regex": "loss=([0-9\\.]+)."},
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
        base_job_name=training_job_name,
        sagemaker_session=sagemaker_session,
        hyperparameters=hp,
        metric_definitions=training_metric_definitions,
    )

    from sagemaker.tuner import HyperparameterTuner
    from sagemaker.workflow.steps import TuningStep

    training_job_name = name_from_base("hpo")

    if HPO:
        hp_tuner = HyperparameterTuner(
            estimator,
            amt_metric_definitions["metrics"][0]["Name"],
            hyperparameter_ranges,
            amt_metric_definitions["metrics"],
            max_jobs=max_jobs,
            max_parallel_jobs=max_parallel_jobs,
            objective_type=amt_metric_definitions["type"],
            base_tuning_job_name=training_job_name,
        )

        tuning_step = TuningStep(
            name="transfer-learning-hpo",
            step_args=hp_tuner.fit(
                inputs={
                    "training": TrainingInput(
                        s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[
                            "finetuning"
                        ].S3Output.S3Uri,
                        content_type="application/x-image",
                        s3_data_type="S3Prefix",
                    ),
                },
            ),
            cache_config=cache_config,
            depends_on=[processing_step],
        )
    else:
        training_step = TrainingStep(
            name="transfer-learning",
            step_args=estimator.fit(
                inputs={
                    "training": TrainingInput(
                        s3_data=processing_step.properties.ProcessingOutputConfig.Outputs[
                            "finetuning"
                        ].S3Output.S3Uri,
                        content_type="application/x-image",
                        s3_data_type="S3Prefix",
                    ),
                },
            ),
            cache_config=cache_config,
            depends_on=[processing_step],
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

    from sagemaker.model import Model

    model = Model(
        image_uri=image_uri,
        source_dir=os.path.join(
            os.environ["S3_PROJECT_URI"], "finetuning/sourcedir/sourcedir.tar.gz"
        ),
        model_data=(
            tuning_step.get_top_model_s3_uri(
                top_k=0, s3_bucket=os.environ["S3_BUCKET_NAME"]
            )
            if HPO
            else training_step.properties.ModelArtifacts.S3ModelArtifacts
        ),
        entry_point="inference.py",
        role=os.environ["SM_EXEC_ROLE"],
        name="tf-birds-detection-finetuned-model",
        sagemaker_session=sagemaker_session,
    )

    from sagemaker.workflow.model_step import ModelStep

    model_step = ModelStep(
        name="create-model",
        step_args=model.create(instance_type=os.environ["PROCESSING_INSTANCE_TYPE"]),
        depends_on=[tuning_step if HPO else training_step],
    )

    # transformer step for evaluation
    from sagemaker.transformer import Transformer
    from sagemaker.workflow.steps import TransformStep
    from sagemaker.workflow.functions import Join

    transformer = Transformer(
        model_name=model_step.properties.ModelName,
        instance_type=os.environ["TRANSFORM_INSTANCE_TYPE"],
        instance_count=int(os.environ["TRANSFORM_INSTANCE_COUNT"]),
        strategy="MultiRecord",
        accept="application/x-image",
        assemble_with="Line",
        output_path=os.environ["S3_TRANSFORM_OUTPUT_URI"],
        base_transform_job_name="tf-od-transformer",
        sagemaker_session=sagemaker_session,
    )
    transform_step = TransformStep(
        name="transform-images",
        display_name="transform images",
        description="This step passes test images to the finetuned model to make predictions (object detections).",
        step_args=transformer.transform(
            job_name="tf-od-transformer",
            data=Join(
                on="/",
                values=[
                    processing_step.properties.ProcessingOutputConfig.Outputs[
                        "validation"
                    ].S3Output.S3Uri,
                    "images/",
                ],
            ),
            split_type="None",
            join_source="None",
            content_type="application/x-image",
        ),
        cache_config=cache_config,
    )

    # evaluation step
    sklearnprocessor = ScriptProcessor(
        image_uri="482497089777.dkr.ecr.eu-north-1.amazonaws.com/cometml:latest",
        role=os.environ["SM_EXEC_ROLE"],
        instance_count=int(os.environ["PROCESSING_INSTANCE_COUNT"]),
        instance_type=os.environ["PROCESSING_INSTANCE_TYPE"],
        env={
            "COMET_API_KEY": os.environ["COMET_API_KEY"],
            "COMET_PROJECT_NAME": os.environ["COMET_PROJECT_NAME"],
            "PC_BASE_DIR": os.environ["PC_BASE_DIR"],
            "TRAINING_JOB_NAME": (
                tuning_step.properties.BestTrainingJob.TrainingJobName
                if HPO
                else training_step.properties.TrainingJobName
            ),
        },
        command=["python3"],
    )

    eval_logging_step = ProcessingStep(
        name="evaluate-and-log",
        processor=sklearnprocessor,
        display_name="evaluate and log",
        description="This step evaluates finetuned model and then logs results using cometml.",
        code="src/evaluation_and_logging.py",
        cache_config=cache_config,
        depends_on=[transform_step, tuning_step if HPO else training_step],
        inputs=[
            ProcessingInput(
                source=os.environ["S3_TRANSFORM_OUTPUT_URI"],
                destination="/opt/ml/processing/input/predictions",
            ),
            ProcessingInput(
                source=Join(
                    on="/",
                    values=[
                        processing_step.properties.ProcessingOutputConfig.Outputs[
                            "validation"
                        ].S3Output.S3Uri,
                        "annotations.json",
                    ],
                ),
                destination="/opt/ml/processing/input/ground_truth",
            ),
            ProcessingInput(
                source=(
                    tuning_step.get_top_model_s3_uri(
                        top_k=0, s3_bucket=os.environ["S3_BUCKET_NAME"]
                    )
                    if HPO
                    else training_step.properties.ModelArtifacts.S3ModelArtifacts
                ),
                destination="/opt/ml/processing/input/model",
            ),
        ],
    )

    # build the pipeline
    pipeline = Pipeline(
        name="tf-birds-detection-pipeline",
        steps=[
            processing_step,
            tuning_step if HPO else training_step,
            model_step,
            transform_step,
            eval_logging_step,
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
