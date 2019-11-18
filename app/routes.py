import os
from collections import defaultdict

from flask import render_template, request, jsonify
from flask_socketio import emit
from . import detector

from app import app


@app.route("/")
@app.route("/index")
def index():
    return "HELLO WORLD!"


@app.route("/analyze", methods=["POST"])
def analyze():
    output = detector.get_boxes(request.data)
    print (output)
    return jsonify(output)