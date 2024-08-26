from flask import Flask, render_template, request
import numpy as np
import base64
import os

from fastai.vision.all import *
    
# Load the fastai learner
learn = load_learner("aedesModel_2.pkl", cpu=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_on_image(image_stream):
    image_stream.seek(0)  # اطمینان از خواندن از ابتدای جریان
    img = PILImage.create(image_stream)
    prediction, idx, probs = learn.predict(img)
    accuracy = int(probs[idx] * 100)
    return prediction, accuracy

def encode_image(image_stream):
    image_stream.seek(0)
    image_data = image_stream.read()
    encoded_image = base64.b64encode(image_data).decode('utf-8')
    return encoded_image

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file and allowed_file(file.filename):
            try:
                encoded_image = encode_image(file.stream)
                file.stream.seek(0)  # بازگرداندن جریان به ابتدای خود برای پیش‌بینی
                prediction, accuracy = predict_on_image(file.stream)
                return render_template('result.html', detection_img_data=encoded_image, prediction=prediction, accuracy=accuracy)
            except Exception as e:
                return render_template('index.html', error=f'Error processing file: {e}')

    return render_template('index.html')

if __name__ == '__main__':
    os.environ.setdefault('FLASK_ENV', 'development')
    app.run(debug=False, port=80, host='0.0.0.0')