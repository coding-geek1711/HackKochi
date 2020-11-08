import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, json, render_template
import os
import shutil
import cv2
from CreateMaskForImage import readImage, createInitialMask, erosionOfMask, findContours
from CreateMaskForImage import findIndividualGrains, drawBoundingRects
import tensorflow as tf
from fullStarter import starter

app = Flask(__name__)

# load Model on startup
FLAG = 0
model = ""


def loadModel():
    model_to_be_loaded = "utils/seed_classifier.h5"
    model = tf.keras.models.load_model(model_to_be_loaded)
    return model


def createTemporaryStorageForImage(file):
    if os.path.exists("temp"):
        shutil.rmtree("temp")
    os.mkdir("temp")
    file.save(f"temp/{file.filename}")
    return "success"


@app.route("/", methods=['GET', 'POST'])
def home():
    if os.path.exists("test_images"):
        shutil.rmtree("test_images")
        shutil.rmtree("temp")
    global FLAG, model
    if FLAG == 0:
        model = loadModel()
    FLAG += 1
    if request.method == 'GET':
        return render_template("index.html")

    if request.method == "POST":
        print(request.files['image'])
        response = createTemporaryStorageForImage(request.files['image'])
        print(response)
        image = starter(model)
        cv2.imshow("IMage", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return jsonify("Success")


app.run(port=4000, debug=True)
