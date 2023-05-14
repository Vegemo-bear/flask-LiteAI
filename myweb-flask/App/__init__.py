from flask import Flask
from flask_cors import CORS
from .views import blue
from .exts import init_exts


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'abcdefghijklmm'

    cors = CORS(app, resources={r"/*/*": {"origins": "*"}})

    app.register_blueprint(blueprint=blue)

    db_uri = 'mysql+pymysql://root:123456@localhost:3306/myweb'
    app.config['SQLALCHEMY_DATABASE_URI'] = db_uri
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    init_exts(app=app)

    return app

