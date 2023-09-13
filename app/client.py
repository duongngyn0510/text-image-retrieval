import base64
import json
from io import BytesIO

import requests
from PIL import Image

api = "http://localhost:30000/image/"
image_file = "images/woman_blazers.png"

image = open(image_file, "rb")
files = {"file": image}
print(files)
response = requests.post(api, files=files)
print(response)

# if response.status_code == 200:
#     data = response.json()
#     encoded_images = data.get("images", [])
#     for encoded_image in encoded_images:
#         image_data = base64.b64decode(encoded_image)
#         if encoded_image:
#             img = Image.open(BytesIO(image_data))
#             img.show()
# else:
#     print("Error:", response.status_code)
