import numpy as np
import tensorflow as tf
from typing import List

class Model(tf.keras.Model):
    def __init__(self, inputs, outputs, sentence_encoder):
        super(Model, self).__init__(inputs, outputs)
        self.sentence_encoder = sentence_encoder
        
    def summarize(self, document:List[str], max_length:int=3):
        embedding = self.sentence_encoder(document).numpy()[None,:,:]
        outputs = self(embedding)[:,:,0]
        selected = np.argsort(-outputs)[:,:max_length]
        selected = [i in selected for i in range(len(document))]
        summary = [document[idx] for idx, i in enumerate(selected) if i]
        return summary, selected


def create_model(sentence_encoder):
    inputs = tf.keras.Input((None, 512))
    x = tf.keras.layers.Dense(128, activation='relu')(inputs)
    x = tf.keras.layers.Conv1D(128, 5, 1, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(128, 5, 1, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(128, 5, 1, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(128, 5, 1, padding='same', activation='relu')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs, sentence_encoder=sentence_encoder)
    model.compile(
        loss = 'binary_crossentropy',
        metrics = ['acc', tf.keras.metrics.Precision(name='Precision'), tf.keras.metrics.Recall(name='Recall')],
        optimizer = tf.keras.optimizers.Adam()
    )
    return model

def CNN(sentence_encoder):
    return create_model(sentence_encoder)