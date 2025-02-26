from dotenv import load_dotenv
import os
from sagemaker import model_uris, image_uris
import subprocess
import tempfile
import shutil
from sagemaker.model import Model
from sagemaker.predictor import Predictor


def run_pipeline():
    load_dotenv(override=True)

    # create a zipped folder with the pre-trained model artifacts
    model_uri = model_uris.retrieve(
        model_id=os.environ["MODEL_ID"],
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
            os.environ["S3_PRETRAINED_MODEL_URI"],
            "--quiet",
        ],
        check=True,
    )
    source_dir = os.path.join(model_dir, "code")

    # create sagemaker model instance of the pre-trained model
    image_uri = image_uris.retrieve(
        region=os.environ["AWS_REGION"],
        framework=None,
        image_scope="inference",
        model_id=os.environ["MODEL_ID"],
        model_version=os.environ["MODEL_VERSION"],
        instance_type=os.environ["INFERENCE_INSTANCE_TYPE"],
    )

    model = Model(
        image_uri=image_uri,
        source_dir=source_dir,
        model_data=os.environ["S3_PRETRAINED_MODEL_URI"],
        entry_point="inference.py",
        role=os.environ["SM_EXEC_ROLE"],
        predictor_cls=Predictor,
        name="tf-birds-detection-model",
        code_location=os.environ["S3_PRETRAINED_OUTPUT_URI"],
    )

    model.deploy(
        initial_instance_count=int(os.environ["INFERENCE_INSTANCE_COUNT"]),
        instance_type=os.environ["INFERENCE_INSTANCE_TYPE"],
        endpoint_name=os.environ["ENDPOINT_NAME"],
        endpoint_logging=True,
    ),

    shutil.rmtree(temp_dir)


if __name__ == "__main__":
    run_pipeline()
