# Author: Haoran Chen
# Date: 2019-09-27

import tensorflow as tf
import pickle
import numpy as np
from pprint import pprint
from collections import defaultdict
from config import Config
from model import SGRU
from gru import GRU
import sys
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor


def get_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # config.gpu_options.visible_device_list = '%d'%(flags.gpu)
    return config


def get_train_op(model, options, global_step):
    lr = tf.train.exponential_decay(options.lr, global_step, 1000, options.wd)
    optimizer = tf.train.AdamOptimizer(lr)
    trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    grads, variables = zip(*optimizer.compute_gradients(model.loss, trainable_variables))
    grads, global_norm = tf.clip_by_global_norm(grads, 40)
    train_op = optimizer.apply_gradients(zip(grads, variables), global_step)
    return train_op


def get_batch1(indices, data_dict):
    '''
    Args:
    indices: sentence ids
    data_dict: dictionary for data

    Return:
    tags: features for tagging, shape (batch_size, tag_dim)
    vid_feats: features for videos, shape (batch_size, video_dim)
    captions: indices representations for sentences which has shape (seqlen, batch_size)
    '''
    train_data = data_dict['train_data']
    eco_res_feat = data_dict['eco_res_feat']
    tag_feat = data_dict['tag_feat']
    max_len = max([len(train_data[0][idx]) for idx in indices])
    captions = np.zeros(shape=(max_len, len(indices)), dtype=np.int32)
    tags, vid_feats = [], []
    for idx1, idx2 in enumerate(indices):
        sent = train_data[0][idx2]
        captions[:len(sent), idx1] = sent
        vid_idx = train_data[1][idx2]
        tags.append(tag_feat[vid_idx])
        vid_feats.append(eco_res_feat[vid_idx])
    tags = np.stack(tags, axis=0)
    vid_feats = np.stack(vid_feats, axis=0)
    return tags, vid_feats, captions


def get_batch2(indices, data_dict, size_per_vid):
    '''
    Args: 
    indices: video ids
    data_dict: dictionary for data
    size_per_vid: number of sentences for each video

    Return:
    tags: features for tagging, shape (batch_size, tag_dim)
    vid_feats: features for video, shape (batch_size, video_dim)
    captions: number representations for sentences 
            which has shape (seqlen, batch_size*size_per_video)
    '''
    idx2gts = data_dict['idx2gts']
    eco_res_feat = data_dict['eco_res_feat']
    tag_feat = data_dict['tag_feat']
    captions = {i: idx2gts[i] for i in indices}
    for key in captions:
        sents = captions[key]
        choices = np.random.choice(np.arange(len(sents)), size_per_vid, False)
        captions[key] = [sents[c] for c in choices]
    max_len = 0
    for key in captions:
        for sent in captions[key]:
            max_len = max(max_len, len(sent))
    captions_np = np.zeros((max_len, len(indices)*size_per_vid), dtype=np.int32)
    for idx1, idx2 in enumerate(indices):
        for idx3, sent in enumerate(captions[idx2]):
            captions_np[:len(sent), idx1*size_per_vid+idx3] = sent
    tags, vid_feats = [], []
    for idx1, idx2 in enumerate(indices):
        tag_f = np.tile(np.expand_dims(tag_feat[idx2], 0), (size_per_vid, 1))
        vid_f = np.tile(np.expand_dims(eco_res_feat[idx2], 0), (size_per_vid, 1))
        tags.append(tag_f)
        vid_feats.append(vid_f)
    tags = np.concatenate(tags, axis=0)
    vid_feats = np.concatenate(vid_feats, axis=0)
    return tags, vid_feats, captions_np


def get_data(flags):
    eco_res_feat = np.load(flags.ecores)
    tag_feat = np.load(flags.tag)
    with open(flags.corpus, 'rb') as fo:
        corpus = pickle.load(fo)
    with open(flags.ref, 'rb') as fo:
        ref = pickle.load(fo)
    idx2word = corpus[4]
    train_data = corpus[0]

    idx2gts = defaultdict(list)
    for sent, vidx in zip(*corpus[0]):
        idx2gts[vidx].append(sent)
    for sent, vidx in zip(*corpus[1]):
        idx2gts[vidx].append(sent)
    for sent, vidx in zip(*corpus[2]):
        idx2gts[vidx].append(sent)

    train_gt_sents = [[idx2word[w] for w in sent] for sent in train_data[0]]
    data_dict = {'train_data': train_data, 'train_gt_sents': train_gt_sents, 
                 'eco_res_feat': eco_res_feat, 'tag_feat': tag_feat, 
                 'idx2word': idx2word, 'corpus': corpus, 
                 'idx2gts': idx2gts, 'ref': ref}
    return data_dict


def get_model(options):
    model = SGRU(options)
    return model


def get_gru(options):
    model = GRU(options)
    return model
    

def get_options(data_dict):
    options = Config(data_dict['corpus'][5])
    return options


def score(ref, hypo):
    """
    ref, dictionary of reference sentences (id, sentence)
    hypo, dictionary of hypothesis sentences (id, sentence)
    score, dictionary of scores
    """
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(),"METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]
    final_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(ref, hypo)
        if type(score) == list:
            for m, s in zip(method, score):
                final_scores[m] = s
        else:
            final_scores[method] = score
    return final_scores


def train_part1(train_idx, train_op, train_loss, 
                sess, options, data_dict, model):
    size_per_vid = 1
    for idx in range(0, options.train_size, options.batch_size):
        start_idx = idx
        end_idx = min(options.train_size, start_idx + options.batch_size)
        tags, vid_feats, captions = get_batch1(
            train_idx[start_idx:end_idx], data_dict)
        feed_dict = {model.word_idx: captions, 
                     model.vid_inputs: vid_feats, 
                     model.se_inputs: tags, 
                     model.size_per_vid: size_per_vid}
        run_ops = {'prob_dist': model.prob_dist, 
                   'loss': model.loss, 
                   'train_op': train_op, 
                   'vid2hid': model.v2h}
        res = sess.run(run_ops, feed_dict)
        train_loss.append(res['loss'])


def train_part2(train_indices, train_op, train_loss, sess, 
                epoch_idx, options, data_dict, model):
    size_per_vid = int(2**(epoch_idx // 16))
    vid_num = options.batch_size // size_per_vid
    for idx in range(0, options.train_size2, vid_num):
        start_idx = idx
        end_idx = min(idx + vid_num, options.train_size2)
        tags, vid_feats, captions = get_batch2(
            train_indices[start_idx:end_idx], data_dict, size_per_vid)
        feed_dict = {model.word_idx: captions, 
                     model.vid_inputs: vid_feats, 
                     model.se_inputs: tags, 
                     model.size_per_vid: size_per_vid}
        run_ops = {'prob_dist': model.prob_dist, 
                   'loss': model.loss, 
                   'train_op': train_op}
        res = sess.run(run_ops, feed_dict)
        train_loss.append(res['loss'])
