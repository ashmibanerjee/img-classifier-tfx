from src.app.tf_serve import *
from src.utils.test_images import image_links

IMAGE_LINKS = image_links

if __name__ == "__main__":
    for IMAGE_LINK in IMAGE_LINKS:
        res = serve_rest(IMAGE_LINK["url"])
        print(res)
