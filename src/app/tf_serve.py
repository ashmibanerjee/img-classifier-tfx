import json
import requests
from src.pred.tf_pred import *

URL = 'http://localhost:9001/v1/models/1657646009:predict'


def write_file(data, file_name="data.json"):
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def serve_rest(img_url):
    img = preprocess_image(img_url)
    data = json.dumps({"signature_name": "serving_default", "instances": img.tolist()})
    write_file(data)
    headers = {"content-type": "application/json"}
    json_response = requests.post(URL, data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    results = decode_predictions(predictions)
    return results
