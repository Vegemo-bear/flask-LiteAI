# ☘️ flask-LiteAI | 后端服务器 | 算法部署平台 ☘️
### ❄️ 技术栈-后端
|  名称   | 版本  |
|  ----  | ----  |
| python  | 3.8.16 |
| flask  | 2.2.3 |
| mysql  | every |
##
### ❄️ 技术栈-算法
|  名称   | 了解  |
|  ----  | ----  |
| OCR  | paddleocr-2.6 |
| Inference  | onnxruntime/tensorRT |
| onnx  | Inference file format |
| Algorithm  | DB + SVTR |
##
### ❄️ 部署流程
1）进入项目目录，终端执行命令（自动安装完项目依赖）：<br>
  pip3 install -r requirements.txt <br>
2）需要自行安装mysql数据库（网上很多教程） <br>
3）生成数据库相关表，依次执行以下命令（中途会生成migrations文件夹）：<br>
  ① flask db init <br>
  ② flask db migrate <br>
  ③ flask db upgrade <br>
##
### ❄️ 项目目录介绍
![图片1](https://github.com/Vegemo-bear/flask-Vue3-LiteAI/assets/127828066/b4f1d747-cd83-4129-8735-ef6393af2c33)
