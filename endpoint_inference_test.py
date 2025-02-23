import sagemaker
import dotenv
from sagemaker import get_execution_role
from sagemaker.predictor import Predictor
from sagemaker.serializers import IdentitySerializer
from sagemaker.deserializers import JSONDeserializer
import os
import subprocess
import tempfile
import tarfile
import json
from pathlib import Path


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


def parse_response(query_response):
    # with open("query_response.json", "w") as outfile:
    #     json.dump(model_predictions, outfile)

    normalized_boxes, classes, scores = (
        query_response["normalized_boxes"],
        query_response["classes"],
        query_response["scores"],
    )
    # take predictions with scores >= 0.5 only
    high_confidence_mask = [score >= 0.5 for score in scores]
    normalized_boxes = [
        box for box, keep in zip(normalized_boxes, high_confidence_mask) if keep
    ]
    classes = [cls for cls, keep in zip(classes, high_confidence_mask) if keep]
    scores = [score for score, keep in zip(scores, high_confidence_mask) if keep]

    temp_dir = tempfile.mkdtemp()
    model_dir = os.path.join(temp_dir, "model.tar.gz")
    subprocess.run(
            ["aws", "s3", "cp", os.environ["S3_MODEL_URI"], model_dir, "--quiet"],
            check=True,
        )
    
    unzipped_model_dir = Path(temp_dir) / "model"
    unzipped_model_dir.mkdir(exist_ok=True)
    with tarfile.open(model_dir, "r:gz") as tar:
        tar.extractall(unzipped_model_dir)

    with open(os.path.join(temp_dir, "model", "labels_info.json"), "r") as f:
        labels_info = json.load(f)
    
    labels = [labels_info["labels"][int(cls)] for cls in classes]

    return normalized_boxes, classes, labels, scores


def run_test():
    dotenv.load_dotenv()
    # SageMaker session setup
    role = os.environ["SM_EXEC_ROLE"]
    sagemaker_session = sagemaker.Session()

    # Replace with your endpoint name
    predictor = Predictor(
        endpoint_name=os.environ["ENDPOINT_NAME"],
        sagemaker_session=sagemaker_session,
        serializer=IdentitySerializer("image/jpeg"),
        deserializer=JSONDeserializer(),
    )

    image = "Black_Footed_Albatross_0032_796115.jpg"
    response = query(predictor, image)

    nbb, cls, lb, scr = parse_response(response)
    print(nbb)


if __name__ == "__main__":
    run_test()
