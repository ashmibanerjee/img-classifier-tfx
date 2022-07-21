import json
import requests

URL = 'http://localhost:9001/v1/models/1657646009:predict'


def serve_rest(img):
    data = json.dumps({"signature_name": "serving_default", "instances": img.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post(URL, data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    return predictions
