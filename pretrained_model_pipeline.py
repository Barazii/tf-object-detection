from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from dotenv import load_dotenv
import os
from sagemaker.workflow.steps import CacheConfig
from sagemaker import model_uris, image_uris
import subprocess
import tempfile
import shutil
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.workflow.model_step import ModelStep


def run_pipeline():
    load_dotenv()

    # create a zipped folder with the pre-trained model artifacts
    model_uri = model_uris.retrieve(
        model_id=os.environ["MODEL_NAME"],
        model_version=os.environ["MODEL_VERSION"],
        model_scope="inference",
    )
    temp_dir = tempfile.mkdtemp()
    model_dir = os.path.join(temp_dir, "model")
    os.makedirs(model_dir, exist_ok=True)

    subprocess.run(
        ["aws", "s3", "cp", "--recursive", model_uri, model_dir, "--quiet"],
        check=True,
    )
    subprocess.run(
        [
            "tar",
            "-czf",
            os.path.join(temp_dir, "model.tar.gz"),
            "-C",
            model_dir,
            ".",
        ],
        check=True,
    )
    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            os.path.join(temp_dir, "model.tar.gz"),
            f"s3://{os.environ['S3_BUCKET_NAME']}/model/model.tar.gz",
            "--quiet",
        ],
        check=True,
    )
    source_dir = os.path.join(model_dir, "code")
    shutil.make_archive(os.path.join(temp_dir, "sourcedir"), "gztar", source_dir)
    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            os.path.join(temp_dir, "sourcedir.tar.gz"),
            f"s3://{os.environ['S3_BUCKET_NAME']}/sourcedir/sourcedir.tar.gz",
            "--quiet",
        ],
        check=True,
    )
    shutil.rmtree(temp_dir)

    # create sagemaker model instance of the pre-trained model
    image_uri = image_uris.retrieve(
        region=os.environ["AWS_REGION"],
        framework=None,
        image_scope="inference",
        model_id=os.environ["MODEL_NAME"],
        model_version=os.environ["MODEL_VERSION"],
        instance_type=os.environ["INFERENCE_INSTANCE_TYPE"],
    )

    model = Model(
        image_uri=image_uri,
        source_dir=os.environ["S3_SOURCEDIR_URI"],
        model_data=os.environ["S3_MODEL_URI"],
        entry_point="inference.py",
        role=os.environ["SM_EXEC_ROLE"],
        predictor_cls=Predictor,
        name="tf-birds-detection-model",
    )

    model.deploy(
        initial_instance_count=int(os.environ["INFERENCE_INSTANCE_COUNT"]),
        instance_type=os.environ["INFERENCE_INSTANCE_TYPE"],
        endpoint_name=os.environ["ENDPOINT_NAME"],
        endpoint_logging=True,
    ),


if __name__ == "__main__":
    run_pipeline()
