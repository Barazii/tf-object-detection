from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep
from dotenv import load_dotenv
from sagemaker.processing import ProcessingInput, ProcessingOutput
import os
from sagemaker.workflow.steps import CacheConfig
from sagemaker.processing import ScriptProcessor
from sagemaker.workflow.pipeline import Pipeline


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

    # build the pipeline
    pipeline = Pipeline(
        name="tf-birds-detection-pipeline",
        steps=[
            processing_step,
        ],
        sagemaker_session=sagemaker_session,
    )

    pipeline.upsert(
        role_arn=os.environ["SM_EXEC_ROLE"],
        description="A pipeline to finetune a tf ic model on the birds dataset.",
    )

    pipeline.start()


if __name__ == "__main__":
    run_pipeline()
