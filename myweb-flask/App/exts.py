from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_jwt_extended import JWTManager

db = SQLAlchemy()
migrate = Migrate()
jwt = JWTManager()


def init_exts(app):
    db.init_app(app=app)
    jwt.init_app(app=app)
    migrate.init_app(app=app, db=db)
