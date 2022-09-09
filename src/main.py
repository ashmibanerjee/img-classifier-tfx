from src.app.tf_serve import *
from src.utils.test_images import image_links
import os

IMAGE_LINKS = image_links
MODEL_LOCATION = os.getcwd() + "/src/models/"

'''
You can also set the following as environment variables
'''
# STAGE = "train"
STAGE = "pre-train"

METHOD = "rest"
#METHOD = "gRPC"

if __name__ == "__main__":
    if not is_dir_exists(MODEL_LOCATION) :
        # Create a new directory because it does not exist
        os.makedirs(MODEL_LOCATION)
    if is_directory_empty(MODEL_LOCATION) or STAGE == "train":
        # Load and save the model when pre-loaded model does not exist
        save_model()

    try:
        for IMAGE_LINK in IMAGE_LINKS:
            if METHOD == "rest":
                res = serve_rest(IMAGE_LINK["url"])
            else:
                res = serve_grpc(IMAGE_LINK["url"])
            print(res)
    except Exception as e:
        print("Server not running \n")
        print(e)
