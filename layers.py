# -*- coding: utf-8 -*-
'''
Author: Haoran Chen
Date: 9/26/2019
'''

import tensorflow as tf
from pprint import pprint

global_kwargs = {
    'initializer': tf.glorot_normal_initializer(),
    'dtype': tf.float32,
    # 'regularizer': tf.keras.regularizers.l2(1e-5)
}

class LayerNormalization(tf.layers.Layer):
    """Applies layer normalization."""

    def __init__(self, hidden_size, layer_name):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_name = layer_name

    def build(self, input_shape):
        with tf.variable_scope("%s_layer_norm" % self.layer_name):
            self.scale = tf.get_variable("layer_norm_scale", [self.hidden_size], 
              initializer=tf.ones_initializer())
            self.bias = tf.get_variable("layer_norm_bias", [self.hidden_size], 
              initializer=tf.zeros_initializer())
        self.built = True

    def call(self, x, epsilon=1e-8):
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
        return norm_x * self.scale + self.bias
        # return x


class EnsembleLayer(tf.layers.Layer):
    def __init__(self, d1, d2, layer_name):
        super().__init__()
        self.d1 = d1
        self.d2 = d2
        self.layer_name = layer_name

    def build(self, shape):
        # print(self.layer_name, shape)
        kwargs = {
            'use_bias': False, 
            'kernel_initializer': tf.glorot_normal_initializer(),
            # 'kernel_regularizer': tf.keras.regularizers.l2(1e-3),
        }
        with tf.variable_scope(self.layer_name):
            self.l1 = tf.layers.Dense(self.d1, **kwargs, name="1")
            self.l2 = tf.layers.Dense(self.d1, **kwargs, name="2")
            self.l3 = tf.layers.Dense(self.d2, **kwargs, name="3")

        self.built = True

    def call(self, input1, input2):
        # tmp tensor shape (batch_size, n_f)
        tmp = tf.multiply(self.l1(input1), self.l2(input2))
        # tmp tensor shape (batch_size, n_h)
        tmp = self.l3(tmp)
        return tmp


class EmbeddingSharedWeights(tf.layers.Layer):
    """Calculate input embeddings and pre-softmax linear with shared weights."""
    def __init__(self, options):
        """Specify characteristic parameters of embedding layer.
        shared weights is the pre-trained parameters from Glove which is not trainable
        and e2h is trainable matrix which perform projecting function.
        Args:
            options: Settings for the layer.
        """
        super().__init__()
        self.embed = options.embed
        self.n_v = options.n_v
        self.n_w = options.n_w
        self.n_h = options.n_h
        self.trainable = options.we_trainable

    def build(self, _):
        with tf.variable_scope("embedding_and_softmax", reuse=tf.AUTO_REUSE):
            shared_weights = tf.constant(self.embed, tf.float32, (self.n_v, self.n_w))
            self.shared_weights = tf.convert_to_tensor(shared_weights)
            self.e2h = tf.get_variable(
                "embed2hidden", (self.n_w, self.n_h), **global_kwargs)
        self.built = True

    def call(self, x):
        """Got token embeddings of x.
        Args: 
            x: An int32 tensor with shape (seqlen, vid_size*size_per_vid)
        Return:
            embeddings: An float32 tensor with shape 
            (seqlen, vid_size*size_per_vid, hidden_dim).
        """
        with tf.name_scope("embedding"):
            embeddings = tf.nn.embedding_lookup(self.shared_weights, x)
            embeddings = tf.tensordot(embeddings, self.e2h, [[-1], [0]])
        return embeddings

    def linear(self, x):
        """Computes logits by running x through a linear layer.
        Args:
            x: a float32 tensor with shape
            (seqlen, vid_size*size_per_vid, hid_dim)
        Return:
            prob_dist: a float32 tensor with shape 
            (seqlen, vid_size*size_per_vid, vocabulary_size)
        """
        with tf.name_scope("pre-softmax_linear"):
            logits = tf.tensordot(x, tf.transpose(self.e2h), [[-1], [0]])
            logits = tf.tensordot(logits, tf.transpose(self.shared_weights), [[-1], [0]])
        return logits


class Layer1(tf.layers.Layer):
    def __init__(self, d1, d2, layer_name):
        super().__init__()
        self.d1 = d1
        self.d2 = d2
        self.layer_name = layer_name

    def build(self, shape):
        # print(self.layer_name, shape)
        with tf.variable_scope(self.layer_name):
            self.wlayer = EnsembleLayer(self.d1, self.d2, "W")
            self.ulayer = EnsembleLayer(self.d1, self.d2, "U")
            self.vlayer = EnsembleLayer(self.d1, self.d2, "V")
            self.ln = LayerNormalization(self.d2, "layer1")
        self.built = True

    def call(self, s, x, h, v):
        '''
        s semantic tensor, shape (batch_size, semantic_dim)
        x input step tensor, shape (batch_size, hidden_dim)
        h hidden state at the last step, shape (batch_size, hidden_dim)
        v video feature, shape (batch_size, video_dim)
        idx: step number
        '''
        # tmp tensor shape (batch_size, hidden_dim)
        tmp = self.wlayer(s, x) + self.ulayer(s, h) + self.vlayer(s, v)
        tmp = self.ln(tmp)
        tmp = tf.sigmoid(tmp)
        return tmp


class Layer2(tf.layers.Layer):
    def __init__(self, d1, d2, layer_name):
        super().__init__()
        self.d1 = d1
        self.d2 = d2
        self.layer_name = layer_name

    def build(self, shape):
        # print(self.layer_name, shape)
        with tf.variable_scope(self.layer_name):
            self.wlayer = EnsembleLayer(self.d1, self.d2, "W")
            self.ulayer = EnsembleLayer(self.d1, self.d2, "U")
            self.vlayer = EnsembleLayer(self.d1, self.d2, "V")
            self.ln = LayerNormalization(self.d2, "layer2")
        self.built = True

    def call(self, s, x, h, v, r):
        '''
        s tensor shape (batch_size, semantic_dim)
        x tensor shape (batch_size, n_h)
        h tensor shape at the last step (batch_size, hidden_dim)
        v tensor shape (batch_size, video_dim)
        r tensor shape (batch_size, hidden_dim)
        '''
        tmp = self.wlayer(s, x) + r*self.ulayer(s, h) + self.vlayer(s, v)
        tmp = self.ln(tmp)
        return tf.tanh(tmp)



if __name__ == "__main__":
    x = tf.random_uniform([4, 4])
    y = tf.random_uniform([4, 4])

    dropout = Dropout(0.5)

    x_d = dropout(x)
    y_d = dropout(y)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    res = sess.run([x, x_d, y, y_d])
    pprint(res)
    res2 = sess.run([x, x_d, y, y_d])
    pprint(res2)
