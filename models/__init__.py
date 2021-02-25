import os
import tensorflow_hub as hub
from .mmr import MMR
from .bilstm import BiLSTM
from .cnn import CNN


def load_sentence_encoder(base_dir):
    model_path = os.path.join(base_dir, 'muse_3')
    model_path = model_path if os.path.isdir(model_path) else "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
    model = hub.load(model_path)
    print(f"Load model successfully from {model_path}!")
    return model

def load_model(model_name, base_dir):
    sentence_encoder = load_sentence_encoder(base_dir)

    if model_name == 'mmr':
        model = MMR(sentence_encoder)
    elif model_name == 'bilstm':
        model = BiLSTM(sentence_encoder)
        model.load_weights(os.path.join(base_dir, 'weights/BiLSTM.h5'))
    elif model_name == 'cnn':
        model = CNN(sentence_encoder)
        model.load_weights(os.path.join(base_dir, 'weights/CNN.h5'))
    
    return model