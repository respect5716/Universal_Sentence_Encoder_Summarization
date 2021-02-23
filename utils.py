import os
import tensorflow_hub as hub


def load_model(base_dir):
    model_path = os.path.join(base_dir, 'universal-sentence-encoder_4')
    model_path = model_path if os.path.isdir(model_path) else "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(model_path)
    print(f"Load model successfully from {model_path}!")
    return model