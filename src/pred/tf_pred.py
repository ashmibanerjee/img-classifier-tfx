import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import time

from src.utils.utilities import *
from src.app.tf_serve import *

SAVE_LOCATION = os.getcwd() + "/resources/"
IMAGE_SHAPE = (224, 224)


def load_model():
    mobilenet_v2 = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
    classifier_model = mobilenet_v2

    classifier = tf.keras.Sequential([
        hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE + (3,))
    ])
    return classifier


def save_model():
    model = load_model()
    ts = int(time.time())
    base_file_path = os.getcwd() + "/models/"
    print(base_file_path)

    file_path = base_file_path + str(ts)
    model.save(filepath=file_path, save_format='tf')


def load_labels():
    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',
                                          'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels'
                                          '.txt')
    imagenet_labels = np.array(open(labels_path).read().splitlines())
    return imagenet_labels


def preprocess_img(img):
    img = img.resize(IMAGE_SHAPE)
    img = np.array(img) / 255.0
    return img


def post_process_results(result):
    predicted_class = tf.math.argmax(result[0], axis=-1)
    scores = tf.nn.softmax(result[0])
    probability = np.max(scores)

    imagenet_labels = load_labels()
    predicted_class_name = imagenet_labels[predicted_class]

    return {"predicted_label": predicted_class_name,
            "probability": probability.item()}


def tf_predict(img_url, serve_type="REST"):
    _img = load_image(img_url)
    img = preprocess_img(_img)
    if serve_type == "REST":
        pred = serve_rest(img[np.newaxis, ...])
    result = post_process_results(pred)
    return result


