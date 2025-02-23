import sagemaker
import dotenv
from sagemaker import get_execution_role
from sagemaker.predictor import Predictor
from sagemaker.serializers import IdentitySerializer
from sagemaker.deserializers import JSONDeserializer
import os
import json


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
    model_predictions = json.loads(query_response)
    with open("query_response.json", "w") as outfile:
        json.dump(model_predictions, outfile)

    normalized_boxes, classes, scores = (
        model_predictions["normalized_boxes"],
        model_predictions["classes"],
        model_predictions["scores"],
    )
    # Filter predictions with scores >= 0.5
    high_confidence_mask = [score >= 0.5 for score in scores]
    normalized_boxes = [
        box for box, keep in zip(normalized_boxes, high_confidence_mask) if keep
    ]
    classes = [cls for cls, keep in zip(classes, high_confidence_mask) if keep]
    scores = [score for score, keep in zip(scores, high_confidence_mask) if keep]

    return normalized_boxes, classes, scores


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

    nbb, cls, scr = parse_response(response)
    print(cls)


if __name__ == "__main__":
    run_test()
