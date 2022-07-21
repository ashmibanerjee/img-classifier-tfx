from src.pred.tf_pred import *
from src.utils.test_images import image_links

IMAGE_LINKS = image_links

if __name__ == "__main__":
    for IMAGE_LINK in IMAGE_LINKS:
        res = tf_predict(IMAGE_LINK["url"], serve_type="REST")
        print(res)
