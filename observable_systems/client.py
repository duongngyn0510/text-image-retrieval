import argparse
import os
import random
import string
import webbrowser
from time import sleep

import requests

base_api = f"http://localhost:8088"


def request_text(text_query):
    text_api = f"{base_api}/display_text"

    headers = {
        "accept": "application/json",
        "content-type": "application/x-www-form-urlencoded",
    }

    params = {"text_query": text_query}

    requests.post(text_api, params=params, headers=headers)


def request_image(image_file):
    image_api = f"{base_api}/display_image"
    image = open(image_file, "rb")
    files = {"image_file": image}

    response = requests.post(image_api, files=files)


if __name__ == "__main__":
    # not in cache

    generated_strings = []
    length = 20
    images_file = os.listdir("/tmp/GLAMI-1M/GLAMI-1M-dataset/images/")
    num_strings_query = 1000

    while len(generated_strings) < num_strings_query:
        characters = string.ascii_lowercase + string.digits
        random_string = "".join(random.sample(characters, length))
        if random_string not in generated_strings:
            generated_strings.append(random_string)

    for i in range(len(images_file)):
        request_image(
            os.path.join("/tmp/GLAMI-1M/GLAMI-1M-dataset/images/", images_file[i])
        )
        request_text(generated_strings[i])
        sleep(1)

    # in cache
    # images_file = os.listdir("/tmp/GLAMI-1M/GLAMI-1M-dataset/images/")
    # for i in range(len(images_file)):
    #     request_image(
    #         os.path.join("/tmp/GLAMI-1M/GLAMI-1M-dataset/images/", images_file[1])
    #     )
    #     sleep(1)
