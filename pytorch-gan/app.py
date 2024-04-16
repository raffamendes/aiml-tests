from flask import Flask, request
import pickle
import torch
import numpy as np

app = Flask(__name__)
# Load model
with open('generator.pkl', 'rb') as f:
    model = pickle.load(f)

model_name = "Handwriting Digits Generator"
model_file = 'generator.plk'
version = "v1.0.0"


@app.route('/info', methods=['GET'])
def info():
    """Return model information, version how to call"""
    result = {}

    result["name"] = model_name
    result["version"] = version

    return result


@app.route('/health', methods=['GET'])
def health():
    """REturn service health"""
    return 'ok'


@app.route('/generate', methods=['GET'])
def predict():
    
    try:
        output = model(torch.randn(1,100))
        output = output.cpu().detach()
        return {
            'status': 200,
            'handwrittendigit': output[0].numpy().tolist()
        }
    except ValueError as e:
        return {'error': str(e).split('\n')[-1].strip()}, 500


if __name__ == '__main__':
    app.run(host='0.0.0.0')