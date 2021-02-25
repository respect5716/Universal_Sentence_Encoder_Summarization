import os
import json
from flask import Flask, request

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_text import SentencepieceTokenizer

from models import load_sentence_encoder, MMR

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default="C:/Users/Administrator/Desktop/Project/Universal_Sentence_Encoder_Summarization")
args = parser.parse_args()

app = Flask(__name__)

sentence_encoder = load_sentence_encoder(args.base_dir)
mmr = MMR(sentence_encoder)
model_dict = {
    'mmr': mmr
}

@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    summary, selected = model_dict[data['model']].summarize(data['document'])
    result = {'summary':summary, 'selected':selected}
    result = json.dumps(result)
    return result


if __name__ == '__main__':
    app.run(port=5001, debug=True)