import os
import json
from flask import Flask, request

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_text import SentencepieceTokenizer

from models import load_model, MMR, BiLSTM

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default="C:/Users/Administrator/Desktop/Project/Universal_Sentence_Encoder_Summarization")
parser.add_argument('--model_name', default='cnn')
args = parser.parse_args()

app = Flask(__name__)
model = load_model(args.model_name, args.base_dir)


@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    summary, selected = model.summarize(data['document'])
    result = {'summary':summary, 'selected':selected}
    result = json.dumps(result)
    return result


if __name__ == '__main__':
    app.run(port=5001, debug=True)