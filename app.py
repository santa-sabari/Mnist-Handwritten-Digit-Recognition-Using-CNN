from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps

app = Flask(__name__)
model = load_model('model.hdf5')

#Image Preprocessing
def prepare_image(image):
    image = ImageOps.grayscale(image)
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    image = np.array(image) / 255
    image = image.reshape(1, 28, 28, 1)
    return image
@app.route('/', methods = ['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = Image.open(file.stream)
            image = prepare_image(image)
            prediction = model.predict(image)
            predicted_class = np.argmax(prediction)
            return render_template('result.html', predicted_class = predicted_class)
    return render_template('index.html')
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)