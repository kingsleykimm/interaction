import base64
import requests
import os
import json


def image_to_base64(images):
    encoded = []
    for image in images:
        with open(os.path.join(os.getcwd(), image), "rb") as image_file:
            # Read the image as binary data
            image_data = image_file.read()
            # Encode the binary data to base64
            base64_encoded = base64.b64encode(image_data)
            # Convert the base64 byte data to a string
            encoded.append(base64_encoded.decode('utf-8'))
    return encoded

imgs = ['8.png', '10.png', '12.png']

encoded_images = image_to_base64(imgs)
response = requests.post("http://localhost:11434/api/generate", data=json.dumps({
    "model" : "llava",
    "prompt" : """These images are going to be a first person view of you as a robot in an environment with a human. 
    Tell me what you see and think is happening from the sequence of images, and what your next steps are.""",
    "stream" : False,
    "images" : encoded_images

}))
print(response.json())