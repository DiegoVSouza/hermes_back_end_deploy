from flask import Blueprint, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
from flask_cors import CORS

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import uuid
from models import Clinica
from models import db

clinica_bp = Blueprint('clinica', __name__)

# Rota para criar uma nova clínica
@clinica_bp.route('/clinica', methods=['POST'])
def create_clinica():
    try:
        data = request.json
        nova_clinica = Clinica(**data)
        db.session.add(nova_clinica)
        db.session.commit()
        return jsonify({'message': 'Clínica criada com sucesso'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@clinica_bp.route('/clinica', methods=['GET'])
def get_clinicas():
    # Obtenha os parâmetros de consulta da URL
    cnpj = request.args.get('cnpj')
    nome = request.args.get('nome')

    # Consulta inicial para todas as clínicas
    query = Clinica.query

    # Filtre a consulta com base nos parâmetros de consulta
    if cnpj:
        query = query.filter(Clinica.cnpj == cnpj)
    if nome:
        query = query.filter(Clinica.nome.ilike(f"%{nome}%"))

    # Execute a consulta
    clinicas = query.all()

    # Converta os resultados em um formato JSON
    clinicas_list = []
    for clinica in clinicas:
        clinicas_list.append({
            'id': clinica.id,
            'cnpj': clinica.cnpj,
            'nome': clinica.nome
        })

    return jsonify(clinicas_list)


# Rota para atualizar os dados de uma clínica
@clinica_bp.route('/clinica/<string:clinica_id>', methods=['PUT'])
def update_clinica(clinica_id):
    try:
        data = request.json
        clinica = Clinica.query.get(clinica_id)
        if not clinica:
            return jsonify({'error': 'Clínica não encontrada'}), 404
        for key, value in data.items():
            setattr(clinica, key, value)
        db.session.commit()
        return jsonify({'message': 'Dados da clínica atualizados com sucesso'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Rota para excluir uma clínica
@clinica_bp.route('/clinica/<string:clinica_id>', methods=['DELETE'])
def delete_clinica(clinica_id):
    try:
        clinica = Clinica.query.get(clinica_id)
        if not clinica:
            return jsonify({'error': 'Clínica não encontrada'}), 404
        db.session.delete(clinica)
        db.session.commit()
        return jsonify({'message': 'Clínica excluída com sucesso'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400