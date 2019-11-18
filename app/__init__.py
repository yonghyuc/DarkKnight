from flask import Flask
from flask_cors import CORS
from flask_restful import Api
from . import *

app = Flask(__name__)
CORS(app)

api = Api(app)