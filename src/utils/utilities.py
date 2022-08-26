from PIL import Image
import os
import requests

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
        print(e)
        print("image could not be opened")


def is_directory_empty(file_path):
    if not os.listdir(file_path):
        return True
    return False
