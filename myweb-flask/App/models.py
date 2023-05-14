from .exts import db


class User(db.Model):
    __tablename__ = 'web_user'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    identity = db.Column(db.String(50))
    username = db.Column(db.String(40), unique=True)
    password = db.Column(db.String(40))
    access_token = db.Column(db.String(400), unique=True)
    resources = db.relationship('Resource', backref='user', lazy=True)

    def __repr__(self):
        return str([self.id, self.identity,self.username, self.password, self.access_token])


class Resource(db.Model):
    __tablename__ = 'web_res'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    res_name = db.Column(db.String(50), unique=True)
    res_id = db.Column(db.Integer, db.ForeignKey(User.id))

