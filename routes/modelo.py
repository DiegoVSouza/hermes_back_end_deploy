from flask import Blueprint, request, jsonify
from models import db, Modelo

modelo_bp = Blueprint('modelo', __name__)

# Rota para criar um novo modelo
@modelo_bp.route('/modelo', methods=['POST'])
def create_modelo():
    try:
        data = request.json
        novo_modelo = Modelo(**data)
        db.session.add(novo_modelo)
        db.session.commit()
        return jsonify({'message': 'Modelo criado com sucesso'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Rota para obter modelos com filtros
@modelo_bp.route('/modelo', methods=['GET'])
def get_modelos():
    # Obtenha os parâmetros de consulta da URL
    id = request.args.get('id')
    nome = request.args.get('nome')
    id_doenca = request.args.get('id_doenca')

    # Consulta inicial para todos os modelos
    query = Modelo.query

    # Filtre a consulta com base nos parâmetros de consulta
    if id:
        query = query.filter(Modelo.id == id)
    if nome:
        query = query.filter(Modelo.nome.ilike(f"%{nome}%"))
    if id_doenca:
        query = query.filter(Modelo.id_doenca == id_doenca)

    # Execute a consulta
    modelos = query.all()

    # Converta os resultados em um formato JSON
    modelos_list = []
    for modelo in modelos:
        modelos_list.append({
            'id': modelo.id,
            'nome': modelo.nome,
            'versao': modelo.versao,
            'acuracia': modelo.acuracia,
            'sensibilidade': modelo.sensibilidade,
            'precisao': modelo.precisao,
            'f1_score': modelo.f1_score,
            'id_doenca': modelo.id_doenca
        })

    return jsonify(modelos_list)


# Rota para atualizar os dados de um modelo
@modelo_bp.route('/modelo/<string:modelo_id>', methods=['PUT'])
def update_modelo(modelo_id):
    try:
        data = request.json
        modelo = Modelo.query.get(modelo_id)
        if not modelo:
            return jsonify({'error': 'Modelo não encontrado'}), 404
        for key, value in data.items():
            setattr(modelo, key, value)
        db.session.commit()
        return jsonify({'message': 'Dados do modelo atualizados com sucesso'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Rota para excluir um modelo
@modelo_bp.route('/modelo/<string:modelo_id>', methods=['DELETE'])
def delete_modelo(modelo_id):
    try:
        modelo = Modelo.query.get(modelo_id)
        if not modelo:
            return jsonify({'error': 'Modelo não encontrado'}), 404
        db.session.delete(modelo)
        db.session.commit()
        return jsonify({'message': 'Modelo excluído com sucesso'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400
