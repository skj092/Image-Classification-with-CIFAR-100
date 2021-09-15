from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
# 
# from keras.models import load_model

# import tensorflow as tf

import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
from tensorflow import keras

from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input, decode_predictions

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
# class_names = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'model.h5'

# Load your trained model

model = models.load_model(MODEL_PATH)   
model.make_predict_function()     
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path):
    img = image.load_img(img_path, target_size=(32, 32))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='caffe')
    model = models.load_model('model.h5')
    preds = model.predict(x)
    score = tf.nn.softmax(preds[0])
    return np.argmax(score)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        preds = model_predict(file_path)
        return preds
    return None


if __name__ == '__main__':
    app.run(debug=True)

