import os
import io
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from PIL import Image

app = Flask(__name__)

MODEL_FILE = 'modelo_simplificado_final.h5'
CLASS_NAMES = ['Vehículo de carga', 'Camioneta', 'Sedan']

# Cargar modelo
model = tf.keras.models.load_model(MODEL_FILE, compile=False)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No se encontró el archivo.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No se seleccionó ningún archivo.'}), 400

    image_bytes = file.read()
    processed_image = preprocess_image(image_bytes)
    prediction = model.predict(processed_image)
    predicted_class_index = int(np.argmax(prediction))
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = float(np.max(prediction))

    return jsonify({
        'class': predicted_class_name,
        'confidence': f"{confidence:.2%}"
    })
