import os
from collections import defaultdict

from flask import render_template
from flask_socketio import emit
from .model.model.config import cfg

from app import app


@app.route("/")
@app.route("/index")
def index():
    return "HELLO WORLD!"


@app.route("/analyze")
def analyze():
    return cfg.DATA_DIR