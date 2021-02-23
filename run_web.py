import json
import requests
import numpy as np
import tensorflow_hub as hub
from nltk.tokenize import sent_tokenize
from flask import Flask, render_template, request

from models import MMR

app = Flask(__name__)
mmr = MMR()

def inference(sentences):
    headers = {'Content-Type':'application/json'}
    address = "http://127.0.0.1:5001/inference"
    data = {'sentences':sentences}

    result = requests.post(address, data=json.dumps(data), headers=headers)
    embeddings = np.array(json.loads(result.content))
    return embeddings

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    document = request.form['document']
    sentences = sent_tokenize(document)
    embeddings = inference(sentences)

    selects = mmr.summarize(embeddings)
    results = [{'sentence':sentence, 'selected':select} for sentence, select in zip(sentences, selects)]
    summaries = [sentences[idx] for idx, i in enumerate(selects) if i]
    return render_template('summary.html', summaries=summaries, results=results)


if __name__ == '__main__':
    app.run(port=5000, debug=True)