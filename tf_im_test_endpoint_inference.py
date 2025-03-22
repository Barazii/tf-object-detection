import sagemaker
import dotenv
from sagemaker.predictor import Predictor
from sagemaker.serializers import IdentitySerializer
from sagemaker.deserializers import JSONDeserializer
import os


def query(model_predictor, image_file_name):
    with open(image_file_name, "rb") as file:
        input_img_rb = file.read()

    query_response = model_predictor.predict(
        input_img_rb,
        {
            "ContentType": "application/x-image",
            "Accept": "application/json;verbose",
        },
    )
    return query_response


def parse_response(query_response):
    probabilities, labels, predicted_label = (
        query_response["probabilities"],
        query_response["labels"],
        query_response["predicted_label"],
    )

    return predicted_label


def run_test():
    dotenv.load_dotenv(override=True)
    sagemaker_session = sagemaker.Session()

    predictor = Predictor(
        endpoint_name=os.environ["ENDPOINT_NAME"],
        sagemaker_session=sagemaker_session,
        serializer=IdentitySerializer("image/jpeg"),
        deserializer=JSONDeserializer(),
    )

    # test images
    input_images_path = "test_input_images"
    output_path = "test_output_images"
    os.makedirs(output_path, exist_ok=True)
    for image in os.listdir(input_images_path):
        if image.lower().endswith((".jpg", ".jpeg")):
            image_path = os.path.join(input_images_path, image)
            response = query(predictor, image_path)
            lbl = parse_response(response)
            print("label: ", lbl)


if __name__ == "__main__":
    run_test()
