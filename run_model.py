import json
import tensorflow_hub as hub
from flask import Flask, request

app = Flask(__name__)

USB = hub.load("C:/Users/Administrator/Desktop/Project/Universal_Sentence_Encoder_Summarization/universal-sentence-encoder_4")

@app.route('/inference', methods=['POST'])
def inference():
    data = request.json
    outputs = USB(data['sentences']).numpy().tolist()
    outputs = json.dumps(outputs)
    return outputs



if __name__ == '__main__':
    app.run(port=5001, debug=True)