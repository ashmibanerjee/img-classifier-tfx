import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import time
import json
import requests

from src.utils.utilities import *
from src.app.tf_serve import *

SAVE_LOCATION = os.getcwd() + "/resources/"
IMAGE_SHAPE = (224, 224)


def load_model():
    classifier_model = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/classification/2"
    classifier = tf.keras.Sequential([
        hub.KerasLayer(classifier_model, input_shape=IMAGE_SHAPE + (3,))
    ])
    print("Model loaded")
    return classifier


def save_model():
    model = load_model()
    ts = int(time.time())
    base_file_path = os.getcwd() + "/src/models/"
    print(base_file_path)

    file_path = base_file_path + str(ts)
    model.save(filepath=file_path, save_format=".tf", signatures=None)
    print(f"Model Saved under {file_path}")


def load_labels():
    labels_path = tf.keras.utils.get_file('ImageNetLabels.txt',
                                          'https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels'
                                          '.txt')
    imagenet_labels = np.array(open(labels_path).read().splitlines())
    return imagenet_labels


def decode_predictions(result):
    predicted_class = tf.math.argmax(result[0], axis=-1)
    scores = tf.nn.softmax(result[0])
    probability = np.max(scores)

    labels = load_labels() # loads image net labels
    predicted_class_name = labels[predicted_class]

    return {"predicted_label": predicted_class_name,
            "probability": probability.item()}


def parse_record(record):
    # depth_major = record.reshape(3, 32, 32)
    # image = np.transpose(depth_major, [1, 2, 0])
    image = np.resize(record,(224,224,3))
    image = np.array(image) / 255.0
    return image[np.newaxis, ...]


def preprocess_cifar_data(file_path):
    x_train, y_train = load_cifar_data(file_path)
    images = []
    for x in x_train[:5]:
        image = parse_record(x)
        images.append(image)
    return images, y_train


def preprocess_image(img_url):
    _img = load_image(img_url)
    if _img is None:
        return None
    img = _img.resize(IMAGE_SHAPE)
    img = np.array(img) / 255.0
    return img[np.newaxis, ...]
