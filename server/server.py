from flask import Flask, request, render_template, url_for, redirect
from model import FlowerClassifier
import torch
import os
import numpy as np
from PIL import Image
import json
from torchvision import datasets, transforms


def data_prep(path):
    img = Image.open(path)
    test_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor()
    ])
    img = test_transform(img)
    return img.unsqueeze(0)


app = Flask(__name__)

model = FlowerClassifier()  # Assuming FlowerClassifier is your model's class
model.load_state_dict(torch.load('flower_classifier.pth'))

classes = ["Daisy", "Rose", "Tulip", "Dandelion", "Sunflower"]


@app.route("/")
def home():
    try:
        pred = request.args.get('pred', '')
        confidence = float(request.args.get('confidence', ''))
    except:
        pred = ""
        confidence = 0

    return render_template("index.html", pred=pred, confidence=int(confidence))



@app.route("/classify", methods=["POST"])
def classify():
    file = request.files["file"]
    try:
        file.save(f"./saved_files/{file.filename}")

        with torch.inference_mode():
            model.eval()
            pred = model(data_prep(f"./saved_files/{file.filename}"))
            pred_class = torch.argmax(torch.softmax(pred, dim=1), dim=1)
            confidence = torch.softmax(pred, dim=1)[0][pred_class] * 100
            print(classes[pred_class])
            pred_class = classes[pred_class]
        os.remove(f"./saved_files/{file.filename}")
        return redirect(url_for('home', pred=pred_class, confidence=confidence.item()))
    except:
        return redirect(url_for('home'))






if __name__ == "__main__":
    app.run(port=5050)
