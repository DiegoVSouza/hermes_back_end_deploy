from flask import Blueprint, request, jsonify
from models import db, Doenca

doenca_bp = Blueprint('doenca', __name__)

# Rota para criar uma nova doença
@doenca_bp.route('/doenca', methods=['POST'])
def create_doenca():
    try:
        data = request.json
        nova_doenca = Doenca(**data)
        db.session.add(nova_doenca)
        db.session.commit()
        return jsonify({'message': 'Doença criada com sucesso'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Rota para obter doenças
@doenca_bp.route('/doenca', methods=['GET'])
def get_doencas():
    # Obtenha os parâmetros de consulta da URL
    id = request.args.get('id')
    nome = request.args.get('nome')

    # Consulta inicial para todas as doenças
    query = Doenca.query

    # Filtre a consulta com base nos parâmetros de consulta
    if id:
        query = query.filter(Doenca.id == id)
    if nome:
        query = query.filter(Doenca.nome.ilike(f"%{nome}%"))

    # Execute a consulta
    doencas = query.all()

    # Converta os resultados em um formato JSON
    doencas_list = []
    for doenca in doencas:
        doencas_list.append({
            'id': doenca.id,
            'nome': doenca.nome
        })

    return jsonify(doencas_list)

# Rota para atualizar os dados de uma doença
@doenca_bp.route('/doenca/<string:doenca_id>', methods=['PUT'])
def update_doenca(doenca_id):
    try:
        data = request.json
        doenca = Doenca.query.get(doenca_id)
        if not doenca:
            return jsonify({'error': 'Doença não encontrada'}), 404
        for key, value in data.items():
            setattr(doenca, key, value)
        db.session.commit()
        return jsonify({'message': 'Dados da doença atualizados com sucesso'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Rota para excluir uma doença
@doenca_bp.route('/doenca/<string:doenca_id>', methods=['DELETE'])
def delete_doenca(doenca_id):
    try:
        doenca = Doenca.query.get(doenca_id)
        if not doenca:
            return jsonify({'error': 'Doença não encontrada'}), 404
        db.session.delete(doenca)
        db.session.commit()
        return jsonify({'message': 'Doença excluída com sucesso'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400