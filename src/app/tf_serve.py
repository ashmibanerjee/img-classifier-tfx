from src.pred.tf_pred import *
import glob
import json
import requests


def get_model_name(file_path):
    model_path = glob.glob(f'{file_path}/*/')[0].split("/")
    model_name = model_path[-2]
    return model_name


MODEL_LOCATION = os.getcwd() + "/src/models/"
URL = f'http://localhost:9001/v1/models/{get_model_name(MODEL_LOCATION)}:predict'


def write_file(data, file_name="data.json"):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def serve_rest(img_url):
    img = preprocess_image(img_url)
    print("Img preprocessing done!")
    data = json.dumps({"signature_name": "serving_default", "instances": img.tolist()})
    # write_file(data)
    headers = {"content-type": "application/json"}
    json_response = requests.post(URL, data=data, headers=headers)
    print("json response received")
    predictions = json.loads(json_response.text)['predictions']
    results = decode_predictions(predictions)
    print("predictions decoded")
    return results
