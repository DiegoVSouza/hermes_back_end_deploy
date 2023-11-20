from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from flask_cors import CORS

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import uuid
from models import db
from routes.pessoa import pessoa_bp
from routes.paciente import paciente_bp
from routes.clinica import clinica_bp
from routes.doenca import doenca_bp
from routes.diagnostico import diagnostico_bp
from routes.funcionario import funcionario_bp
from routes.medico import medico_bp
from routes.modelo import modelo_bp
from models import Pessoa
from flask_jwt_extended import JWTManager, create_access_token, jwt_required

app = Flask(__name__)

CORS(app, origins='*')
CORS(pessoa_bp, origins='*')
app.config['JWT_SECRET_KEY'] = 'hermes123'
jwt = JWTManager(app)

# Carrega o modelo .h5
model1 = tf.keras.models.load_model('./modeloXception.h5')
model2 = tf.keras.models.load_model('./CNN_modelvgg19.h5')
models = []
models.append(model1)
models.append(model2)

db_user = 'postgres'
db_password = '123'
db_host = 'localhost'
db_port = '5432'
db_name = 'hermesdb'

connection_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
app.config['SQLALCHEMY_DATABASE_URI'] = connection_string
db.init_app(app)

app.register_blueprint(pessoa_bp)
app.register_blueprint(paciente_bp)
app.register_blueprint(modelo_bp)
app.register_blueprint(funcionario_bp)
app.register_blueprint(doenca_bp)
app.register_blueprint(diagnostico_bp)
app.register_blueprint(clinica_bp)
app.register_blueprint(medico_bp)


@app.route('/predict/<int:model_id>', methods=['POST'])
def predict(model_id):
    try:
        print(model_id)

        image = request.files['image'].read()


        image = tf.image.decode_image(image, channels=3)
        image = tf.image.resize(image, (224, 224))
        image = tf.expand_dims(image, axis=0) 

        # image = tf.image.rgb_to_grayscale(image)

        image = tf.cast(image, tf.float32) / 255.0
        model = models[model_id - 1]
     
        predictions = model.predict(image)
        predictions = predictions.tolist()
        print(predictions)
        data = {'predictions': predictions}
        response = jsonify(data)
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response
    except Exception as e:
        response = jsonify({'error': str(e)})
        return response

if __name__ == '__main__':
    app.run(debug=True,host="localhost", port=5000)
