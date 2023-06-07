from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

app = Flask(__name__)

CORS(app)

@app.route('/')
def home():
    return "¡La API de predicción está en funcionamiento!"


@app.route('/predict', methods=['POST'])
def predict():

    # Carga el modelo y los pesos
    longitud, altura = 100, 100
    modelo = './modelo/modelo.h5'
    pesos = './modelo/pesos.h5'
    cnn = load_model(modelo)
    cnn.load_weights(pesos)

    # Obtiene la imagen enviada en la solicitud POST
    file = request.files['image']
    x = load_img(file)
    x = x.resize((longitud, altura))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)

    # Realiza la predicción
    arreglo = cnn.predict(x)
    resultado = arreglo[0]
    respuesta = np.argmax(resultado)
    
    # Asigna la etiqueta correspondiente
    if respuesta == 0:
        etiqueta = "Gato"
    elif respuesta == 1:
        etiqueta = "Perro"
    else:
        etiqueta = "Clase no encontrada"

    # Retorna el resultado en formato JSON
    return jsonify({'prediction': etiqueta})


if __name__ == '__main__':
    app.run()
