import sys

import torch
from PIL import Image
from flask import Flask, request
from flask_restful import Resource, Api

from dataset import transform
from predict import predict

app = Flask("pokeball")


@app.route('/', methods=['POST'])
def pokemon_predict():
    try:
        X = []
        for file in request.files.getlist("files"):
            f = file
            image = Image.open(f.stream).convert("RGB")
            image_tensor = transform(image)
            X.append(image_tensor)
        labels = predict(torch.stack(X))
        return {"success": True, "data": labels}
    except:
        e = sys.exc_info()[0]
        print("Unexpected error:", e)
        return {"success": False}


if __name__ == '__main__':
    app.run()
