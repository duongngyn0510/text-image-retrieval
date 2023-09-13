import json


def get_image_url(match_ids):
    with open("./id2url.json", "r") as f:
        id2url = json.load(f)
    images_url = []
    for i in match_ids:
        images_url.append(id2url[i])
    return images_url
