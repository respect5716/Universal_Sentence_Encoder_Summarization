import numpy as np
import tensorflow as tf
from typing import List

def gelu(x):
    cdf = 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(2.0)))
    return x * cdf

class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, seq_len, model_dim):
        super(EmbeddingLayer, self).__init__()
        self.token_embedding = tf.keras.layers.Dense(model_dim, activation='relu')
        self.position_embedding = tf.keras.layers.Embedding(seq_len, model_dim)
        self.pos = tf.range(0, seq_len)

    def call(self, x):
        x = self.token_embedding(x)
        pos = self.position_embedding(self.pos)
        x += pos
        return x

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, model_dim, num_head):
        super(MultiHeadAttention, self).__init__()
        self.model_dim = model_dim
        self.num_head = num_head
        self.projection_dim = self.model_dim // self.num_head
        assert self.model_dim % self.num_head == 0

        self.qw = tf.keras.layers.Dense(self.model_dim)
        self.kw = tf.keras.layers.Dense(self.model_dim)
        self.vw = tf.keras.layers.Dense(self.model_dim)
        self.w = tf.keras.layers.Dense(self.model_dim)
    
    def attention(self, q, k ,v, mask):
        dim = tf.cast(tf.shape(q)[-1], tf.float32)
        score = tf.matmul(q, k, transpose_b=True)
        scaled_score = score / tf.math.sqrt(dim)

        if mask is not None:
            scaled_score += (mask * -1e9)

        attention_weights = tf.nn.softmax(scaled_score)
        attention_outputs = tf.matmul(attention_weights, v)
        return attention_outputs, attention_weights
    
    def split_heads(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_head, self.projection_dim))
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return x
    
    def combine_heads(self, x):
        batch_size = tf.shape(x)[0]
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (batch_size, -1, self.model_dim))
        return x
    
    def call(self, q, k, v, mask):
        q, k, v = self.qw(q), self.kw(k), self.vw(v)
        q, k, v = self.split_heads(q), self.split_heads(k), self.split_heads(v)
        outputs, weights = self.attention(q, k, v, mask)
        outputs = self.combine_heads(outputs)
        outputs = self.w(outputs)
        return outputs

class FeedForwardNetwork(tf.keras.layers.Layer):
    def __init__(self, model_dim, ffn_dim):
        super(FeedForwardNetwork, self).__init__()
        self.dense1 = tf.keras.layers.Dense(ffn_dim)
        self.dense2 = tf.keras.layers.Dense(model_dim)

    def call(self, x):
        x = self.dense1(x)
        x = gelu(x)
        x = self.dense2(x)
        return x

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, num_head, model_dim, ffn_dim, drop_rate):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(model_dim, num_head)
        self.ffn = FeedForwardNetwork(model_dim, ffn_dim)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, enc, training, padding_mask=None):
        out1 = self.mha(enc, enc, enc, padding_mask)
        out1 = self.dropout1(out1, training=training)
        out1 = self.layernorm1(enc + out1)
        out2 = self.ffn(out1)
        out2 = self.dropout2(out2, training=training)
        out2 = self.layernorm2(out1 + out2)
        return out2

class Model(tf.keras.Model):
    def __init__(self, inputs, outputs, sentence_encoder):
        super(Model, self).__init__(inputs, outputs)
        self.sentence_encoder = sentence_encoder
        
    def summarize(self, document:List[str], max_length:int=3):
        padded_document = document + [""] * (100 - len(document))
        embedding = self.sentence_encoder(padded_document).numpy()[None,:,:]
        outputs = self(embedding)[:, :len(document), 0]
        selected = np.argsort(-outputs)[:,:max_length]
        selected = [i in selected for i in range(len(document))]
        summary = [document[idx] for idx, i in enumerate(selected) if i]
        return summary, selected


def create_model(sentence_encoder):
    inputs = tf.keras.Input((None, 512))
    x = EmbeddingLayer(100, 128)(inputs)
    x = EncoderLayer(2, 128, 128, 0.2)(x)
    x = EncoderLayer(2, 128, 128, 0.2)(x)
    x = EncoderLayer(2, 128, 128, 0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = Model(inputs, outputs, sentence_encoder=sentence_encoder)
    model.compile(
        loss = 'binary_crossentropy',
        metrics = ['acc', tf.keras.metrics.Precision(name='Precision'), tf.keras.metrics.Recall(name='Recall')],
        optimizer = tf.keras.optimizers.Adam()
    )
    return model


def Transformer(sentence_encoder):
    return create_model(sentence_encoder)