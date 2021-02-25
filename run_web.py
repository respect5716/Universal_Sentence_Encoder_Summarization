import json
import requests
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

def preprocess(document):
    document = document.split('\n')
    return document

def inference(document):
    headers = {'Content-Type':'application/json'}
    address = "http://127.0.0.1:5001/inference"
    data = {'document':document}

    result = requests.post(address, data=json.dumps(data), headers=headers)
    result = json.loads(result.content)
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    document = request.form['document']
    document = preprocess(document)
    result = inference(document)
    summary, selected = result['summary'], result['selected']
    result = [{'sentence':_sentence, 'selected':_selected} for _sentence, _selected in zip(document, selected)]
    return render_template('summary.html', summary=summary, result=result)


if __name__ == '__main__':
    app.run(port=5000, debug=True)