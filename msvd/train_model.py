# -*- coding: utf-8 -*-
# Author: Haoran Chen
# Date: 2019-09-17
import tensorflow as tf
import pickle
import numpy as np
import sys
from pprint import pprint
from collections import defaultdict
import time
sys.path.append('..')
from utils import *


np.random.seed(42)
data_dict = None
model = None
options = None

METRICS = {'Bleu_4': 0., 'CIDEr': 0., 
           'METEOR': 0., 'ROUGE_L': 0.}
# METRICS = {'ROUGE_L': 0.}
MAX = {key: 0. for key in METRICS}
min_xe = 1000.

def cal_metrics(sess, phase):
    sent_dict, sent_list = defaultdict(list), []
    loss_list = []
    logits_dict = {'xe': [], 'all': []}
    if phase == "train":
        ref = data_dict["ref"][0]
        idx2cap = {idx: elem for idx, elem in enumerate(ref)}
        idx_start, idx_end = 0, 1200
    elif phase == "val":
        ref = data_dict['ref'][1]
        idx2cap = {idx+1200: elem for idx, elem in enumerate(ref)}
        idx_start, idx_end = 1200, 1300
    elif phase == "test":
        ref = data_dict['ref'][2]
        idx2cap = {idx+1300: elem for idx, elem in enumerate(ref)}
        idx_start, idx_end = 1300, 1970
    else:
        raise ValueError("The phase should be val or test")
    tag_feat = data_dict['tag_feat']
    eco_res_feat = data_dict['eco_res_feat']
    idx2gts = data_dict['idx2gts']
    for idx in range(idx_start, idx_end):
        tag, ervid = tag_feat[idx], eco_res_feat[idx]
        tag, ervid = np.expand_dims(tag, 0), np.expand_dims(ervid, 0)
        gts = idx2gts[idx]
        maxlen = max([len(gt) for gt in gts])
        gts_mat = np.zeros((maxlen, len(gts)), dtype=np.int32)
        for idx2, gt in enumerate(gts):
            gts_mat[:len(gt), idx2] = gt
        # print('tag shape:', tag.shape, 'evid:', evid.shape, 'rvid:', rvid.shape)
        wanted_ops = {
            'generated_words': model.generated_words, 'test_loss': model.test_loss, 
            'xe_logits': model.xe_logits, 'all_logits': model.all_logits}
        feed_dict = {
            model.word_idx: gts_mat, model.vid_inputs: ervid, model.se_inputs: tag}
        # sel_word_idx shape: (batch_size, beam_width, n_steps)
        res = sess.run(wanted_ops, feed_dict)
        generated_words = res['generated_words']
        loss_list.append(res['test_loss'])
        logits_dict['xe'].append(res['xe_logits'])
        logits_dict['all'].append(res['all_logits'])
        for x in np.squeeze(generated_words):
            if x == 0:
                break
            sent_dict[idx].append(data_dict['idx2word'][x])
        sent_dict[idx] = [' '.join(sent_dict[idx])]
        sent_list.append(sent_dict[idx][0])
    scores = score(idx2cap, sent_dict)
    print(phase)
    pprint(scores)
    mean_loss = np.mean(loss_list)
    print('average loss:', mean_loss, flush=True)

    with open(flags.name+'_%s_output.log'%phase, 'w') as fo:
        for sent in sent_list:
            fo.write(sent+'\n')
    with open(flags.name+'_%s_logits.pkl'%phase, 'wb') as fo:
        pickle.dump([logits_dict['xe'], logits_dict['all']], fo, -1)
    return scores, mean_loss


def main():
    global data_dict, model, options
    data_dict = get_data(flags)
    options = get_options(data_dict)
    model = get_model(options)
    # model = get_gru(options)
    best_score, save_path = 0., None

    with model.graph.as_default():
        global_step = tf.train.get_or_create_global_step()
        train_op = get_train_op(model, options, global_step)
        saver = tf.train.Saver()
        config = get_config()
        sess = tf.Session(config=config, graph=model.graph)

        if flags.test is None:
            sess.run(tf.global_variables_initializer())
            train_idx1 = np.arange(options.train_size, dtype=np.int32)
            train_idx2 = np.arange(options.train_size2, dtype=np.int32)

            for idx in range(options.epoch):
                start_time = time.perf_counter()
                train_loss = []
                if idx < options.threshold:
                    np.random.shuffle(train_idx1)
                    train_part1(train_idx1, train_op, train_loss, 
                                sess, options, data_dict, model)
                else:
                    np.random.shuffle(train_idx2)
                    train_part2(train_idx2, train_op, train_loss, sess, 
                                idx, options, data_dict, model)
                mean_train_loss = np.mean(train_loss)
                print('epoch %d: loss %f.' % (idx, mean_train_loss))
                scores, mean_val_loss = cal_metrics(sess, 'val')
                # update maximum metrics values
                global METRICS, MAX, min_xe
                METRICS = {key: max(METRICS[key], scores[key]) for key in METRICS}
                overall_score1 = np.mean([scores[key] / METRICS[key] for key in METRICS])
                overall_score2 = np.mean([MAX[key] / METRICS[key] for key in METRICS])
                if overall_score1 > overall_score2:
                    MAX = scores
                    save_path = saver.save(sess, './saves/%s-best.ckpt'%flags.name)
                    print('Epoch %d: the best model has been saved as %s.'
                        % (idx, save_path), flush=True)
                end_time = time.perf_counter()
                print('%d epoch: %.2fs.' % (idx, end_time - start_time))
            saver.restore(sess, save_path)
            cal_metrics(sess, "train")
            cal_metrics(sess, 'test')
        else:
            saver.restore(sess, flags.test)
            cal_metrics(sess, 'train')
            cal_metrics(sess, 'val')
            cal_metrics(sess, 'test')
        sess.close()


if __name__ == "__main__":
    tf.app.flags.DEFINE_string('name', '1', 'name of model')
    tf.app.flags.DEFINE_string('corpus', None, 'Path to corpus file')
    tf.app.flags.DEFINE_string('ecores', None, 'Path to ECO-RES feature files')
    tf.app.flags.DEFINE_string('tag', None, 'Path to Tag feature files')
    tf.app.flags.DEFINE_string('ref', None, 'Path to reference files')
    tf.app.flags.DEFINE_string('test', None, 'Path to the saved parameters')

    flags = tf.app.flags.FLAGS

    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    print('Total time: %.2fs' % (end_time - start_time))
