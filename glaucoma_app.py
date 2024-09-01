from flask import Flask, request, jsonify, send_file
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import io

app = Flask(__name__)

# Load the TensorFlow model
model = tf.keras.models.load_model('my_model2.h5')

def import_and_predict(image_data):
    image = ImageOps.fit(image_data, (100, 100), Image.ANTIALIAS)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = (image.astype(np.float32) / 255.0)
    img_reshape = image[np.newaxis, ...]
    prediction = model.predict(img_reshape)
    return prediction

@app.route('/')
def index():
    return "Welcome to the Glaucoma Detector API!"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.lower().endswith('.jpg'):
        image = Image.open(io.BytesIO(file.read()))
        prediction = import_and_predict(image)
        pred = prediction[0][0]
        if pred > 0.5:
            result = "Your eye is Healthy. Great!"
        else:
            result = "You are affected by Glaucoma. Please consult an ophthalmologist as soon as possible."
        return jsonify({'prediction': result})
    else:
        return jsonify({'error': 'Invalid file format'}), 400

if __name__ == '__main__':
    app.run(port=5000, debug=True)  # You can specify any port you prefer
