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
try:
    model = tf.keras.models.load_model(MODEL_FILE, compile=False)
    print(f"✅ Modelo '{MODEL_FILE}' cargado correctamente.")
except Exception as e:
    print(f"❌ Error al cargar el modelo: {e}")
    model = None

# Preprocesamiento de imagen
def preprocess_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print("❌ Error al procesar la imagen:", e)
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'El modelo no está cargado.'}), 500

        if 'file' not in request.files:
            return jsonify({'error': 'No se encontró el archivo.'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No se seleccionó ningún archivo.'}), 400

        image_bytes = file.read()

        if len(image_bytes) > 5 * 1024 * 1024:
            return jsonify({'error': 'La imagen es demasiado grande (máx. 5MB).'}), 400

        processed_image = preprocess_image(image_bytes)
        if processed_image is None:
            return jsonify({'error': 'No se pudo procesar la imagen.'}), 400

        prediction = model.predict(processed_image)
        predicted_class_index = int(np.argmax(prediction))
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = float(np.max(prediction))

        print("✅ Enviando respuesta JSON al cliente")
        return jsonify({
            'class': predicted_class_name,
            'confidence': f"{confidence:.2%}"
        })

    except Exception as e:
        print(f"❌ Error en la predicción: {e}")
        return jsonify({'error': f'Ocurrió un error: {str(e)}'}), 500
