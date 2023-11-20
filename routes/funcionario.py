from flask import Blueprint, request, jsonify
from models import Funcionario
from models import db

funcionario_bp = Blueprint('funcionario', __name__)

# Rota para criar um novo funcionário
@funcionario_bp.route('/funcionario', methods=['POST'])
def create_funcionario():
    try:
        data = request.json
        novo_funcionario = Funcionario(**data)
        db.session.add(novo_funcionario)
        db.session.commit()
        return jsonify({'message': 'Funcionário criado com sucesso'}), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@funcionario_bp.route('/funcionario', methods=['GET'])
def get_funcionarios():
    # Obtenha os parâmetros de consulta da URL
    id_pessoa = request.args.get('id_pessoa')
    email = request.args.get('email')
    especialista = request.args.get('especialista')
    id_clinica = request.args.get('id_clinica')

    # Consulta inicial para todos os funcionários
    query = Funcionario.query

    # Filtre a consulta com base nos parâmetros de consulta
    if id_pessoa:
        query = query.filter(Funcionario.id_pessoa == id_pessoa)
    if email:
        query = query.filter(Funcionario.email == email)
    if especialista:
        query = query.filter(Funcionario.especialista == especialista)
    if id_clinica:
        query = query.filter(Funcionario.id_clinica == id_clinica)

    # Execute a consulta
    funcionarios = query.all()

    # Converta os resultados em um formato JSON
    funcionarios_list = []
    for funcionario in funcionarios:
        funcionarios_list.append({
            'id': funcionario.id,
            'id_pessoa': funcionario.id_pessoa,
            'email': funcionario.email,
            'especialista': funcionario.especialista,
            'id_clinica': funcionario.id_clinica
        })

    return jsonify(funcionarios_list)


# Rota para atualizar os dados de um funcionário
@funcionario_bp.route('/funcionario/<string:funcionario_id>', methods=['PUT'])
def update_funcionario(funcionario_id):
    try:
        data = request.json
        funcionario = Funcionario.query.get(funcionario_id)
        if not funcionario:
            return jsonify({'error': 'Funcionário não encontrado'}), 404
        for key, value in data.items():
            setattr(funcionario, key, value)
        db.session.commit()
        return jsonify({'message': 'Dados do funcionário atualizados com sucesso'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Rota para excluir um funcionário
@funcionario_bp.route('/funcionario/<string:funcionario_id>', methods=['DELETE'])
def delete_funcionario(funcionario_id):
    try:
        funcionario = Funcionario.query.get(funcionario_id)
        if not funcionario:
            return jsonify({'error': 'Funcionário não encontrado'}), 404
        db.session.delete(funcionario)
        db.session.commit()
        return jsonify({'message': 'Funcionário excluído com sucesso'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400