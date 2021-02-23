import os
import json
import tensorflow_hub as hub
from flask import Flask, request

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default="C:/Users/Administrator/Desktop/Project/Universal_Sentence_Encoder_Summarization")
args = parser.parse_args()

app = Flask(__name__)

def load_model(base_dir):
    model_path = os.path.join(base_dir, 'universal-sentence-encoder_4')
    model_path = model_path if os.path.isdir(model_path) else "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(model_path)
    return model


@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    outputs = model(data['sentences']).numpy().tolist()
    outputs = json.dumps(outputs)
    return outputs


if __name__ == '__main__':
    model = load_model(args.base_dir)
    app.run(port=5001, debug=True)