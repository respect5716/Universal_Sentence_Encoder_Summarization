import os
import json
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask, request

from utils import load_model

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default="C:/Users/Administrator/Desktop/Project/Universal_Sentence_Encoder_Summarization")
args = parser.parse_args()

app = Flask(__name__)


@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    with tf.device('/cpu:0'):
        outputs = model(data['sentences']).numpy().tolist()
    outputs = json.dumps(outputs)
    return outputs


if __name__ == '__main__':
    model = load_model(args.base_dir)
    app.run(port=5001, debug=True)