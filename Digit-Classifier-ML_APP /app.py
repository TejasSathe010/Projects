from flask import Flask, render_template, request
from scipy.misc import imsave, imread, imresize
import numpy as np
import keras.models
import re

import base64

import sys
import os

sys.path.append(os.path.abspath("./model"))
from load import *

app = Flask(__name__)
global model
model = init()


#decoding an image from base64 into raw representation
def convertImage(imgData1):
    imgstr = re.search(b'base64,(.*)', imgData1).group(1)
    with open('output.png', 'wb') as output:
        output.write(base64.b64decode(imgstr))


@app.route('/')
def index():
    # initModel()
    return render_template("index.html")


@app.route('/predict/', methods=['GET', 'POST'])
def predict():
    imgData = request.get_data()
    # print(imgData)
    convertImage(imgData)

    x = imread('output.png', mode='L')
    x = np.invert(x)
    x = imresize(x, (28, 28))
    # imshow(x)
    x = x.reshape(1, 28, 28, 1)
    out = model.predict(x)
    print(out)
    print(np.argmax(out, axis=1))
    response = np.array_str(np.argmax(out, axis=1))
    response = response[1]
    return response


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    # app.run(host='0.0.0.0', port=port)
    app.run(debug=True)