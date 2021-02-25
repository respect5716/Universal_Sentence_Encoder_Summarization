import os
import tensorflow_hub as hub
from .mmr import MMR
from .bilstm import BiLSTM


def load_sentence_encoder(base_dir):
    model_path = os.path.join(base_dir, 'universal-sentence-encoder-multilingual_3')
    model_path = model_path if os.path.isdir(model_path) else "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3"
    model = hub.load(model_path)
    print(f"Load model successfully from {model_path}!")
    return model