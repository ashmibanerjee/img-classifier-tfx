from PIL import Image
import os
import requests
import pickle
import numpy as np
from pathlib import Path

current_dir = os.getcwd()
os.chdir(current_dir + "/../")


def is_dir_exists(dir_location):
    isExist = os.path.exists(dir_location)
    if not isExist:
        return False
    else:
        return True


def load_image(img_url):
    try:
        img = Image.open(requests.get(img_url, stream=True).raw)
        return img
    except Exception as e:
        try:
            img = Image.open(img_url)
            return img
        except Exception as e:
            print(e)
            print("image could not be opened")


def load_cifar_data(file_name):
    for i in range(1, 6):
        dictionary = unpickle_data(file_name + str(i))
        x_data = dictionary[b'data']
        y_data = np.array(dictionary[b"labels"])
        if i == 1:
            x_train = x_data
            y_train = y_data
        else:
            x_train = np.concatenate((x_train, x_data), axis=0)
            y_train = np.concatenate((y_train, y_data), axis=0)
    return x_train, y_train


def is_directory_empty(file_path):
    if not os.listdir(file_path):
        return True
    return False


def unpickle_data(file_name):
    print("file name", file_name)
    with open(file_name, "rb") as fo:
        data = pickle.load(fo, encoding="bytes")
    return data


def rename_folder(current_name: str,
                  new_name: str,
                  path: Path) -> None:
    os.rename(os.path.join(path, current_name), os.path.join(path, new_name))
