from flask import Blueprint, jsonify, request
from .models import *
from flask_jwt_extended import create_access_token
import os, base64
from datetime import datetime
from werkzeug.utils import secure_filename
from App.lpdr.lpdr import lp_img_preprocess, lp_refer, lp_postprocess, lp_generate_result
from App.ppocrv3.ppocrv3 import pp_img_preprocess, pp_refer, pp_postprocess, pp_generate_result

blue = Blueprint('user', __name__)


@blue.route('/')
def test():
    return '服务器可用'


@blue.route('/test/', methods=['GET'])
def index():
    return jsonify({
        'code': 0,
        'data': {
            'data': "flask + vue3 成功连通（跨域）！"
        }
    })


@blue.route('/users/login/', methods=['GET', 'POST'])
def user_login():
    vue_username = request.form.get('username')
    vue_password = request.form.get('password')

    u = User()
    flask_username = list(User.query.filter(User.username == vue_username))
    flask_password = list(User.query.filter(User.username == vue_username).filter(User.password == vue_password))

    res = 'vx.jpg'
    img_data = open(os.path.join('App/static/img/resource/others/', str(res)), "rb").read()
    img_data = base64.b64encode(img_data).decode('utf-8')

    if flask_username and flask_password:
        flask_username = flask_username[0].username
        access_token = flask_password[0].access_token
        flask_identity = flask_password[0].identity
        token = access_token
        res = jsonify({
            "success": True,
            "state": 1,
            "message": "登录成功",
            "content": {
                "access_token": token,
                "token_type": "string",
                "img_data": img_data,
                "username": flask_username,
                "identity": flask_identity
            }
        })
        return res
    else:
        if vue_username == 'vegemo-bear':
            vue_identity = '超级管理员'
            u.identity = vue_identity
        else:
            vue_identity = '普通用户'
            u.identity = vue_identity
        u.username = vue_username
        u.password = vue_password
        token = create_access_token(identity=vue_username)
        u.access_token = token
        try:
            db.session.add(u)
            db.session.commit()
            res = jsonify({
                "success": True,
                "state": 1,
                "message": "登录成功",
                "content": {
                    "access_token": token,
                    "token_type": "string",
                    "img_data": img_data,
                    "username": vue_username,
                    "identity": vue_identity
                }
            })
            return res
        except:
            db.session.rollback()
            db.session.flush()
            res = jsonify({
                'success': False,
                'state': 0,
                'message': '登录失败',
                "content": {
                    "access_token": 'null',
                    "token_type": "null"
                }
            })
            return res


@blue.route('/users/logout/', methods=['POST'])
def user_logout():
    res = jsonify({
        'success': True,
        'state': 1,
        'message': '退出成功',
        "content": {
            "access_token": '不需要返回',
            "token_type": "不需要返回"
        }
    })
    return res


@blue.route('/users/getAll/', methods=['GET'])
def user_getAll():
    now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    arr = []
    for i in range(len(User.query.all())):
        temp = []
        user_id = User.query.all()[i].id
        temp.append(user_id)
        user_identity = User.query.all()[i].identity
        temp.append(user_identity)
        user_username = User.query.all()[i].username
        temp.append(user_username)
        user_password = User.query.all()[i].password
        temp.append(user_password)
        access_token = User.query.all()[i].access_token
        temp.append(access_token)
        arr.append(temp)

    res = []
    for i in range(len(arr)):
        tp = {
            'id': arr[i][0],
            'identity': arr[i][1],
            'username': arr[i][2],
            'password': arr[i][3],
            'token': arr[i][4]
        }
        res.append(tp)

    result = jsonify({
        'code': '000000',
        'data': res,
        'message': '处理成功',
        'time': now_time
    })
    return result


@blue.route('/users/delete/<string:id>', methods=['DELETE'])
def user_del(id):
    u = User.query.get(id)
    now_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    try:
        db.session.delete(u)
        db.session.commit()
        result = jsonify({
            'code': '000000',
            'data': True,
            'message': '处理成功',
            'time': now_time
        })
        return result
    except:
        db.session.rollback()
        db.session.flush()
        result = jsonify({
            'code': '111111',
            'data': False,
            'message': '处理失败',
            'time': now_time
        })
        return result


@blue.route('/users/upload/', methods=['GET', 'POST'])
def user_upload():
    rec_ = []
    if request.method == 'POST':
        file = request.files.get('file')
        if file is not None:
            filename = secure_filename(file.filename)
            data = request.form.get('data')
            img_path = 'App/static/img/resource/' + str(data) + '/' + filename
            file.save(img_path)
            infer_result = 'App/static/img/dest/' + str(data) + '/result_' + filename
            if data == 'lpdr':
                try:
                    img = open(img_path, 'rb').read()
                    pre_img, ratio_h, ratio_w, src_h, src_w = lp_img_preprocess(img)
                    prob = lp_refer(pre_img)
                    post_result = lp_postprocess(prob, ratio_h, ratio_w, src_h, src_w)
                    rec_res = lp_generate_result(img_path, post_result, infer_result)
                    for i in range(len(rec_res)):
                        rec_.append(rec_res[i][0])
                    img_data = open(infer_result, "rb").read()
                    img_data = base64.b64encode(img_data).decode('utf-8')
                    res = jsonify({
                        "success": True,
                        "state": 1,
                        "message": "推理成功",
                        "content": {
                            "img_data": img_data,
                            'rec_result': rec_
                        }
                    })
                    return res
                except:
                    res = jsonify({
                        "success": False,
                        "state": 0,
                        "message": "图片中有效目标为零",
                        "content": {
                            "img_data": 'null',
                        }
                    })
                    return res
            if data == 'ppocrv3':
                try:
                    img = open(img_path, 'rb').read()
                    pre_img, ratio_h, ratio_w, src_h, src_w = pp_img_preprocess(img)
                    prob = pp_refer(pre_img)
                    post_result = pp_postprocess(prob, ratio_h, ratio_w, src_h, src_w)
                    rec_res = pp_generate_result(img_path, post_result, infer_result)
                    for i in range(len(rec_res)):
                        rec_.append(rec_res[i][0])
                    img_data = open(infer_result, "rb").read()
                    img_data = base64.b64encode(img_data).decode('utf-8')
                    res = jsonify({
                        "success": True,
                        "state": 1,
                        "message": "推理成功",
                        "content": {
                            "img_data": img_data,
                            'rec_result': rec_
                        }
                    })
                    return res
                except:
                    res = jsonify({
                        "success": False,
                        "state": 0,
                        "message": "图片中有效目标为零",
                        "content": {
                            "img_data": 'null',
                        }
                    })
                    return res
        else:
            res = jsonify({
                "success": False,
                "state": 0,
                "message": "后端接收不到图片",
                "content": {
                    "img_data": 'null',
                }
            })
            return res
    else:
        res = jsonify({
            "success": False,
            "state": 0,
            "message": "请求方法应为POST",
            "content": {
                "img_data": 'null',
            }
        })
        return res
