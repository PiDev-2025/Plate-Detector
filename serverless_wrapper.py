from flask import Flask, request, jsonify, Response
import os
import json

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': 'Tunisian License Plate Detection API',
        'message': 'This is a proxy service. For full functionality, please use the Docker container.'
    })

@app.route('/detect_plate', methods=['POST'])
def detect_plate_proxy():
    # For Vercel, we'll return a message directing to use Docker
    # since the actual model is too large for serverless
    return jsonify({
        'message': 'This is a serverless proxy. For full functionality, please use the Docker container or hosted API.',
        'error': 'Model cannot be loaded in serverless environment',
        'success': False
    }), 501

# Vercel serverless handler
def handler(request):
    with app.test_client() as client:
        return client.open(
            request['path'],
            method=request['method'],
            headers=request['headers'],
            data=request.get('body', ''),
            environ_base={'REMOTE_ADDR': request.get('ip', '')}
        )