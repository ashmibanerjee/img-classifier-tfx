from src.pred.tf_pred import *
import glob
import json, os
import requests

import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import tensorflow as tf

from src.pred.tf_pred import preprocess_image

REST_API_PORT = "9001"
GRPC_PORT = "9000"


def get_model_name(file_path):
    model_path = glob.glob(f'{file_path}/*/')[0].split("/")
    model_name = model_path[-2]
    return model_name


MODEL_LOCATION = os.getcwd() + "/src/models/"


def write_file(data, file_name="data.json"):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def make_rest_api_call(img, rest_api_url):
    data = json.dumps({"signature_name": "serving_default", "instances": img.tolist()})
    # write_file(data)
    headers = {"content-type": "application/json"}
    json_response = requests.post(rest_api_url, data=data, headers=headers)
    try:
        predictions = json.loads(json_response.text)['predictions']
        return predictions
    except Exception as e:
        print(e)
        return None


def serve_rest_cifar(file_path, rest_api_url=None):
    if rest_api_url is None:
        try:
            rest_api_url = f'http://localhost:{REST_API_PORT}/v1/models/{get_model_name(MODEL_LOCATION)}:predict'
        except Exception as e:
            print(e)
    images, y_train = preprocess_cifar_data(file_path)
    results = []
    for image in images:
        prediction = make_rest_api_call(image, rest_api_url)
        result = decode_predictions(prediction)
        print(result)
        results.append(result)
    return results


def serve_rest(img_url, rest_api_url=None):
    if rest_api_url is None:
        try:
            rest_api_url = f'http://localhost:{REST_API_PORT}/v1/models/{get_model_name(MODEL_LOCATION)}:predict'
        except Exception as e:
            print(e)

    img = preprocess_image(img_url)
    predictions = make_rest_api_call(img, rest_api_url)
    results = decode_predictions(predictions)
    # print("predictions decoded")
    return results


GRPC_MAX_RECEIVE_MESSAGE_LENGTH = 4096 * 4096 * 3  # Max LENGTH the GRPC should handle


def make_grpc_req(img, model_name, grpc_url, mode="one-by-one"):
    channel = grpc.insecure_channel(grpc_url,
                                    options=[('grpc.max_receive_message_length', GRPC_MAX_RECEIVE_MESSAGE_LENGTH)])

    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    grpc_request = predict_pb2.PredictRequest()
    grpc_request.model_spec.name = model_name
    grpc_request.model_spec.signature_name = 'serving_default'

    if mode == "one-by-one":
        grpc_request.inputs['keras_layer_input'].CopyFrom(tf.make_tensor_proto(img.tolist(), shape=img.shape))
    else:
        print(f"type of img {type(img)}")
        grpc_request.inputs['keras_layer_input'].CopyFrom(tf.make_tensor_proto(img, shape=[len(img)]))
    predictions = stub.Predict(grpc_request, 10)

    # converting from tensor proto to numpy

    outputs_tensor_proto = predictions.outputs["keras_layer"]
    shape = tf.TensorShape(outputs_tensor_proto.tensor_shape)
    outputs = np.array(outputs_tensor_proto.float_val).reshape(shape.as_list())
    return outputs


def serve_grpc(img_url, model_name=None, grpc_url=None, mode="one-by-one"):
    if not isinstance(img_url, str):  # when input is already preprocessed
        # print("image is already preprocessed")
        img = img_url
    else:
        print("we preprocess image")
        img = preprocess_image(img_url)
    if grpc_url is None:
        grpc_url = f"localhost:{GRPC_PORT}"
    if model_name is None:
        model_name = get_model_name(MODEL_LOCATION)

    outputs = make_grpc_req(img, model_name, grpc_url)
    # decoding predictions
    results = decode_predictions(outputs)
    # print("predictions decoded")
    return results
