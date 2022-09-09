from src.pred.tf_pred import *
import glob
import json, os
import requests

import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import tensorflow as tf

REST_API_PORT = "9001"
GRPC_PORT = "9000"


def get_model_name(file_path):
    model_path = glob.glob(f'{file_path}/*/')[0].split("/")
    model_name = model_path[-2]
    return model_name


MODEL_LOCATION = os.getcwd() + "/src/models/"
try:
    REST_API_URL = f'http://localhost:{REST_API_PORT}/v1/models/{get_model_name(MODEL_LOCATION)}:predict'
except Exception as e:
    print(e)


def write_file(data, file_name="data.json"):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def serve_rest(img_url):
    img = preprocess_image(img_url)
    print("Img preprocessing done!")
    data = json.dumps({"signature_name": "serving_default", "instances": img.tolist()})
    # write_file(data)
    headers = {"content-type": "application/json"}
    json_response = requests.post(REST_API_URL, data=data, headers=headers)
    print("json response received")
    predictions = json.loads(json_response.text)['predictions']
    results = decode_predictions(predictions)
    print("predictions decoded")
    return results


GRPC_MAX_RECEIVE_MESSAGE_LENGTH = 4096 * 4096 * 3  # Max LENGTH the GRPC should handle


def serve_grpc(img_url):
    channel = grpc.insecure_channel(f'localhost:{GRPC_PORT}',
                                    options=[('grpc.max_receive_message_length', GRPC_MAX_RECEIVE_MESSAGE_LENGTH)])
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    grpc_request = predict_pb2.PredictRequest()
    grpc_request.model_spec.name = get_model_name(MODEL_LOCATION)
    grpc_request.model_spec.signature_name = 'serving_default'

    img = preprocess_image(img_url)

    grpc_request.inputs['keras_layer_input'].CopyFrom(tf.make_tensor_proto(img.tolist(), shape=img.shape))
    predictions = stub.Predict(grpc_request, 10)

    # converting from tensor proto to numpy

    outputs_tensor_proto = predictions.outputs["keras_layer"]
    shape = tf.TensorShape(outputs_tensor_proto.tensor_shape)
    outputs = np.array(outputs_tensor_proto.float_val).reshape(shape.as_list())

    # decoding predictions
    results = decode_predictions(outputs)
    print("predictions decoded")
    return results
