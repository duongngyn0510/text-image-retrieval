import os
import webbrowser

import requests

# route display_image
api = "http://localhost:30000/display_image"

image_file = "images/bikini.png"
image = open(image_file, "rb")
files = {"image_file": image}
response = requests.post(api, files=files)

if response.status_code == 200:
    with open("temp.html", "w", encoding="utf-8") as f:
        f.write(response.text)
    webbrowser.open("temp.html")

if os.path.exists("temp.html"):
    os.remove("temp.html")
