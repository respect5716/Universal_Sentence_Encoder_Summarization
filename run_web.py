import json
import requests
import numpy as np
from nltk.tokenize import sent_tokenize
from flask import Flask, render_template, request

from models import MMR

app = Flask(__name__)
mmr = MMR()

def preprocess(document):
    document = document.split('\n')
    title = document[0]
    sentence = []
    for i in document[1:]:
        sentence += sent_tokenize(i)
    return title, sentence

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
    title, sentence = preprocess(document)
    embedding = inference(sentence)

    selected = mmr.summarize(embedding)
    result = [{'sentence':_sentence, 'selected':_selected} for _sentence, _selected in zip(sentence, selected)]
    summary = [sentence[idx] for idx, i in enumerate(selected) if i]
    return render_template('summary.html', title=title, summary=summary, result=result)


if __name__ == '__main__':
    app.run(port=5000, debug=True)