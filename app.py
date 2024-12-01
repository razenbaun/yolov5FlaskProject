from flask import Flask, request, render_template, jsonify
import torch
from PIL import Image
import cv2
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

UPLOAD_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file uploaded", 400

    image_file = request.files['image']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
    image_file.save(image_path)

    image = Image.open(image_path)
    results = model(image)
    df = results.pandas().xyxy[0]

    unique_classes = df['name'].unique().tolist()
    return render_template('filter.html', classes=unique_classes, image_path=image_file.filename)


@app.route('/filter', methods=['POST'])
def filter_results():
    selected_classes = request.form.getlist('classes')
    confidence_threshold = float(request.form.get('confidence', 0.5))

    image_path = os.path.join(app.config['UPLOAD_FOLDER'], request.form['image_path'])
    image = Image.open(image_path)
    results = model(image)
    df = results.pandas().xyxy[0]

    filtered_results = df[(df['name'].isin(selected_classes)) & (df['confidence'] >= confidence_threshold)]

    filtered_image = np.array(image)
    for _, row in filtered_results.iterrows():
        label = row['name']
        conf = row['confidence']
        xmin, ymin, xmax, ymax = row[['xmin', 'ymin', 'xmax', 'ymax']]
        color = (0, 255, 0)
        cv2.rectangle(filtered_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 1)
        cv2.putText(filtered_image, f'{label} {conf:.2f}', (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    color, 2)

    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"filtered_{request.form['image_path']}")
    cv2.imwrite(output_path, cv2.cvtColor(filtered_image, cv2.COLOR_RGB2BGR))

    return render_template('rezult.html', image_path=os.path.basename(output_path))


if __name__ == '__main__':
    app.run(debug=True)
