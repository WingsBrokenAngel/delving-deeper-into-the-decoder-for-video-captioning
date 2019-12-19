#-*- coding: utf-8 -*-
'''
Author: Haoran Chen
Initial Date: 11/12/2019
'''

import tensorflow as tf
from layers import *

global_kwargs = {
    'initializer': tf.glorot_normal_initializer(),
    'dtype': tf.float32,
}

class GRU():
    def __init__(self, options):
        '''
        n_w is word embedding dimension.
        n_h is hidden state dimension.
        n_f is mid-input dimension.
        n_v is the size of vocabulary.
        n_t is the dimension of tagging.
        n_z is the total video dimension.
        n_z1 is the ECO dimension.
        n_z2 is the ResNeXt dimension.
        '''
        self.options = options
        self.n_w = options.n_w
        self.n_h = options.n_h
        self.n_f = options.n_f
        self.n_t = options.n_t
        self.n_z = options.n_z
        self.keep_prob = options.keep_prob
        self.gamma = options.gamma

        self.graph = tf.Graph()
        self.layers = []
        with self.graph.as_default():
            tf.set_random_seed(42)
            self.word_idx = tf.placeholder(
                tf.int32, [None, None], name='caption_idx')
            self.vid_inputs = tf.placeholder(
                tf.float32, [None, self.n_z], name='video_inputs')
            self.se_inputs = tf.placeholder(
                tf.float32, [None, self.n_t], name='semantic_inputs')
            self.size_per_vid = tf.placeholder(
                tf.int32, [], name='captions_per_video')
            self.layers.append(tf.keras.layers.GRU(self.n_h, 
                                                   dropout=1-self.keep_prob, 
                                                   recurrent_dropout=1-self.keep_prob, 
                                                   return_sequences=True))
            self.layers.append(tf.keras.layers.GRU(self.n_h, 
                                                   dropout=1-self.keep_prob, 
                                                   recurrent_dropout=1-self.keep_prob, 
                                                   return_sequences=True))

            # self.embed tensor shape (vocabulary_size, word_embed)
            self.embed_layer = EmbeddingSharedWeights(options)
            self.v2h = tf.get_variable("vid2hid", (self.n_z, self.n_h), **global_kwargs)

            self.construct_train_model(
                self.word_idx, self.vid_inputs, self.se_inputs, self.keep_prob)

            self.construct_test_model(self.word_idx, self.vid_inputs, self.se_inputs)

    def construct_train_model(
        self, word_idx, vid_inputs, se_inputs, keep_prob):
        ''' Costruct the model.
        Args:
            word_idx: shape (seqlen, vid_size*size_per_vid)
            vid_inputs: shape (vid_size*size_per_vid, feat_dim)
            se_inputs: shape (vid_size*size_per_vid, semantic_dim)
            keep_prob: keep rate for dropout 
        '''
        # idx_embed shape (seqlen, vid_size*size_per_vid, hidden_dim)
        seqlen = tf.shape(word_idx)[0]
        batch_size_all = tf.shape(word_idx)[1]
        idx_embed = self.embed_layer(word_idx)
        # vid_embed shape (1, vid_size*size_per_vid, hidden_dim)
        vid_embed = tf.expand_dims(vid_inputs @ self.v2h, axis=0)
        # idx_embed shape (seqlen, vid_size*size_per_vid, hidden_dim)
        wlist = tf.concat((vid_embed, idx_embed[:-1]), axis=0)
        wlist = tf.transpose(wlist, (1, 0, 2))
        wlist = tf.nn.dropout(wlist, keep_prob, (batch_size_all, 1, self.n_h))
        # slist shape (vid_size*size_per_vid, seamantic_dim)
        se_inputs = tf.nn.dropout(se_inputs, keep_prob)
        vid_inputs = tf.nn.dropout(vid_inputs, keep_prob)
        init = tf.zeros((batch_size_all, self.n_h))

        # hlist0, hlist1 shape (seqlen, vid_size*size_per_vid, hidden_dim)
        seqlen = tf.shape(wlist)[0]
        
        hlist0 = self.layers[0](wlist, initial_state=init)
        hlist0 = tf.nn.dropout(hlist0, keep_prob, (batch_size_all, 1, self.n_h))
        hlist1 = self.layers[1](hlist0, initial_state=init)
        hlist1 = tf.nn.dropout(hlist1, keep_prob, (batch_size_all, 1, self.n_h))
        hlist1 = tf.transpose(hlist1, (1, 0, 2))
        vid_size = tf.cast(tf.shape(hlist1)[1] / self.size_per_vid, tf.int32)
        # self.prob_dist shape (seqlen, vid_size*size_per_vid, vocabulary_size)
        self.prob_dist = self.embed_layer.linear(hlist1) + 1e-8
        # normalized word probability distribution
        prob_logits = tf.nn.softmax(self.prob_dist)
        # weights shape (seq_len, vid_size*size_per_vid)
        weights = tf.cast(tf.not_equal(word_idx, 0), tf.float32)
        weights = tf.concat(
            (tf.ones((1, batch_size_all), dtype=tf.float32), weights[:-1]), axis=0)
        # lens shape (vid_size*size_per_vid)
        lens = tf.reduce_sum(weights, axis=0)
        weights_modulated = weights / lens
        lens_reshape = tf.reshape(lens, (vid_size, self.size_per_vid))
        lens_logits = tf.nn.softmax(
            -(tf.abs(lens_reshape - self.options.avglen) + 1), -1)
        # xe_loss shape: (seq_len, vid_size*size_per_vid)
        xe_loss = tf.losses.sparse_softmax_cross_entropy(
            word_idx, self.prob_dist, 
            weights_modulated, reduction=tf.losses.Reduction.NONE)

        # xe_loss shape: (vid_size * size_per_vid)
        xe_loss = tf.reduce_sum(xe_loss, 0)
        xe_loss = tf.reshape(xe_loss, (vid_size, self.size_per_vid))
        loss_logits = tf.nn.softmax(-xe_loss, -1)
        loss_logits = loss_logits * self.gamma + lens_logits * (1 - self.gamma)
        self.loss = (tf.reduce_sum(xe_loss * loss_logits) / 
            tf.cast(tf.shape(xe_loss)[0], tf.float32)) #+ self.wd_loss) 

    def construct_test_model(self, word_idx, vid_inputs, se_inputs):
        ''' Costruct the model.
        Args:
            word_idx: shape (seqlen, batch_size)
            vid_inputs: shape (1, feat_dim)
            se_inputs: shape (1, semantic_dim)
        '''
        # vid_embed shape (1, hidden_dim)
        vid_embed = vid_inputs @ self.v2h
        batch_size = tf.shape(word_idx)[1]
        # wlist shape (seqlen, batch_size, hidden_dim)
        init = (tf.zeros((1, self.n_h)), tf.zeros((1, self.n_h)))
        time_steps = tf.range(tf.shape(word_idx)[0], dtype=tf.int32)

        def step(hs, step_idx):
            h0_t_1, h1_t_1 = hs
            # step_prob_dist shape (batch_size, vocabulary_size)
            step_prob_dist = self.embed_layer.linear(h1_t_1)
            # step_word_idx shape (batch_size,)
            step_word_idx = tf.argmax(step_prob_dist, -1)
            # step_word_embed shape (batch_size, hidden_dim)
            step_word_embed = self.embed_layer(step_word_idx)
            x = tf.cond(tf.equal(step_idx, 0), lambda:vid_embed, lambda: step_word_embed)
            x = tf.expand_dims(x, axis=0)
            # h0_t tensor shape (1, batch_size, hidden_dim)
            h0_t = self.layers[0](x, initial_state=h0_t_1)
            h1_t = self.layers[1](h0_t, initial_state=h1_t_1)

            h0_t = tf.reshape(h0_t, (1, self.n_h))
            h1_t = tf.reshape(h1_t, (1, self.n_h))
            return h0_t, h1_t

        # hlist shape ((seqlen, batch_size, hidden_dim), (seqlen, batch_size, hidden_dim))
        hlist = tf.scan(step, time_steps, init)[-1]
        # self.test_prob_dist shape (seqlen, 1, vocabulary_size)
        self.test_prob_dist = self.embed_layer.linear(hlist)
        # test_prob_dist_tile shape (seqlen, batch_size, vocabulary_size)
        test_prob_dist_tile = tf.tile(self.test_prob_dist, (1, batch_size, 1)) + 1e-8
        # generated_words shape (seqlen, batch_size)
        self.generated_words = tf.argmax(self.test_prob_dist, -1)
        # weights shape (seqlen, batch_size)
        weights = tf.cast(tf.not_equal(word_idx, 0), tf.float32)
        weights = tf.concat(
            (tf.ones((1, batch_size), dtype=tf.float32), weights[:-1]), axis=0)
        weights = weights / tf.reduce_sum(weights, axis=0, keepdims=True)
        test_loss = tf.losses.sparse_softmax_cross_entropy(
            word_idx, test_prob_dist_tile, weights, 
            reduction=tf.losses.Reduction.SUM)
        # self.test_wd_loss = tf.losses.get_regularization_loss()
        self.test_loss = test_loss / tf.cast(batch_size, tf.float32) #+ self.test_wd_loss 
            
