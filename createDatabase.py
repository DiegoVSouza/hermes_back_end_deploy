from flask import Flask
from flask_sqlalchemy import SQLAlchemy
import uuid
from models import db
from models import Pessoa, Paciente, Funcionario, Medico, Doenca, Modelo, Diagnostico

# Substitua os valores abaixo pelas suas credenciais e informações de conexão
db_user = 'postgres'
db_password = '123'
db_host = 'localhost'
db_port = '5432'
db_name = 'hermesdb'

# Crie uma string de conexão para o PostgreSQL
connection_string = f'postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = connection_string
db.init_app(app)

with app.app_context():
    db.create_all()



