# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 13:22:59 2017

@author: Jinzitian
"""

import os
import jieba
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

from word2id.word2id import build_dataset
from model.CNN_LSTM import LSTMs_Model



#tf.app.flags.DEFINE_integer('vocabulary_size', 51000, 'vocabulary_size')
#cell_size = 60 #一般取词汇量的 1/400 或 1/550
tf.app.flags.DEFINE_integer('cell_size', 128, 'cell_size')
tf.app.flags.DEFINE_integer('num_layers', 1, 'num_layers')
#样本量由小到大一些的话通常取3种值0.8/0.5/0.35
tf.app.flags.DEFINE_float('keep_prob', 0.8, 'keep_prob')
tf.app.flags.DEFINE_float('init_scale', 0.1, 'init_scale')
tf.app.flags.DEFINE_integer('batch_size', 16, 'batch_size')
tf.app.flags.DEFINE_integer('num_steps', 50, 'num_steps')
tf.app.flags.DEFINE_float('lr_decay', 0.9, 'lr_decay')
tf.app.flags.DEFINE_float('learning_rate', 1.0, 'learning_rate')
tf.app.flags.DEFINE_integer('nb_epoch', 10, 'nb_epoch')
tf.app.flags.DEFINE_string('checkpoints_dir', r'./checkpoints', 'checkpoints save path.')
tf.app.flags.DEFINE_string('model_prefix', 'LSTM_model', 'model save filename.')

FLAGS = tf.app.flags.FLAGS




def get_words2id_data_2():
  
    a = pd.read_excel(r'./data/neg.xls',header=None)
    temp_3 = [(list(jieba.cut(i)),[1,0]) for i in a[0]]
    b = pd.read_excel(r'./data/pos.xls',header=None)
    temp_4 = [(list(jieba.cut(i)),[0,1]) for i in b[0]]
    
    temp_5 = temp_3 + temp_4
    
    with open(r'./data/标点集.txt','r',encoding = 'utf-8') as f:
        characters = f.readlines()
    character_list = [i.split('\n')[0] for i in characters]
    character_dict = {i:0 for i in character_list}
#    stop_words_dict = {}
    words = [[k for k in i if k not in character_dict] for i,j in temp_5]
    tag = [j for i,j in temp_5]
    
    allwords = []
    for i in words:
        allwords.extend(i)

    dictionary, words_tuple = build_dataset(allwords)
    
    word_id = [[dictionary.get(j,0) for j in i] for i in words]
    
    max_length = 0
    for i in word_id:
        if len(i) > max_length:
            max_length = len(i)
            
    word_id_padding = [((i + [0]*(max_length - len(i))) if len(i) < max_length else i) for i in word_id]
        
    return np.array(word_id_padding), np.array(tag), dictionary, words_tuple


def train():

    aa, bb, word2id_dictionary, words_tuple = get_words2id_data_2()
    
    #用pickle保存word2id字典，为了后续预测数据时转换使用
    output = open(r'./dict/word2id_dictionary.pkl', 'wb')
    pickle.dump(word2id_dictionary, output)
    output.close()
    
    output1 = open(r'./dict/words_tuple.pkl', 'wb')
    pickle.dump(words_tuple, output1)
    output1.close()

    #欠采样    
    a_pos = aa[bb.argmax(axis=1) == 1]
    b_pos = bb[bb.argmax(axis=1) == 1]
    a_neg = aa[bb.argmax(axis=1) == 0]
    b_neg = bb[bb.argmax(axis=1) == 0]
        
    a_pos = a_pos[:a_neg.shape[0]]
    b_pos = b_pos[:a_neg.shape[0]]
    aa = np.concatenate((a_pos, a_neg))
    bb = np.concatenate((b_pos, b_neg))
    
    s = np.random.permutation(aa.shape[0])
    aa = aa[s]
    bb = bb[s]    
    position = int(aa.shape[0]*2/3)
    input_train = aa[:position]
    targets_train = bb[:position]
    
    input_test = aa[position:]
    targets_test = bb[position:]

    input_train = input_train[:,:FLAGS.num_steps]
    #计算总的训练批次数
    train_num = input_train.shape[0]
    batch_num = train_num // FLAGS.batch_size
            
    with tf.Graph().as_default():
        #设置整个graph的初始化方式
        initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale)

        x_train = tf.placeholder(tf.int32, shape=[FLAGS.batch_size, FLAGS.num_steps])
        y_train = tf.placeholder(tf.float32, shape=[FLAGS.batch_size, 2])                
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = LSTMs_Model(len(words_tuple), FLAGS.cell_size, FLAGS.num_layers, FLAGS.keep_prob, x_train, y_train)
        
        x_test = input_test[:,:FLAGS.num_steps][:1000]
        test_tag_list = targets_test[:1000]
        with tf.variable_scope("Model", reuse=True, initializer=initializer):
            m_test = LSTMs_Model(len(words_tuple), FLAGS.cell_size, FLAGS.num_layers, FLAGS.keep_prob, x_test)
            y_pre_tf = tf.argmax(m_test.predict,1)

        saver = tf.train.Saver()
            
        with tf.Session() as session:
            session.run(tf.global_variables_initializer()) 
         
            for j in range(FLAGS.nb_epoch): 
                S = 0
                n = 0 
                for i in range(batch_num):
                    lr_decay_new = max(FLAGS.lr_decay ** max(i-1, 0.0)*0.01, 0.001)
                    session.run(m.lr_update, feed_dict = {m.new_lr: FLAGS.learning_rate * lr_decay_new})
                    _, cost = session.run([m.train_op,m.cost], feed_dict = {x_train: input_train[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size], y_train: targets_train[i*FLAGS.batch_size:(i+1)*FLAGS.batch_size]})
                    S += cost
                    n += 1                    
                    if i%100 == 0 and i > 0:
                        print("Epoch: %d batch_num: %d loss: %.3f" % (j, i, S/n))
                        S = 0
                        n = 0

            y_pre = session.run(y_pre_tf)
            y_tag = np.array(list(map(np.argmax, test_tag_list)))
            accuracy = np.mean(y_pre == y_tag)
            print('accuracy : ', accuracy)
            #保存模型时一定注意保存的路径必须是英文的，中文会报错
#            save_path = saver.save(session, os.path.join(FLAGS.checkpoints_dir, FLAGS.model_prefix))
            save_path = saver.save(session, FLAGS.checkpoints_dir + '/'+ FLAGS.model_prefix)
            print("Model saved in file: ", save_path)



def main(argv):
    
    train()
    

if __name__ == '__main__':
    
    
    tf.app.run()

    