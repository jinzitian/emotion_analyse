# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 13:22:59 2017

@author: Jinzitian
"""

#import os
import jieba
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

from word2id.word2id import build_dataset
from model.VAR_LSTM import LSTMs_Model



#tf.app.flags.DEFINE_integer('vocabulary_size', 51000, 'vocabulary_size')
#lstm_cell_size = 60 #一般取词汇量的 1/400 或 1/550
tf.app.flags.DEFINE_integer('lstm_cell_size', 128, 'lstm_cell_size')
tf.app.flags.DEFINE_integer('lstm_num_layers', 2, 'lstm_num_layers')
#样本量由小到大一些的话通常取3种值0.8/0.5/0.35
tf.app.flags.DEFINE_float('lstm_keep_prob', 0.8, 'lstm_keep_prob')
tf.app.flags.DEFINE_float('lstm_init_scale', 0.1, 'lstm_init_scale')
tf.app.flags.DEFINE_integer('lstm_batch_size', 16, 'lstm_batch_size')
tf.app.flags.DEFINE_float('lstm_lr_decay', 0.9, 'lstm_lr_decay')
tf.app.flags.DEFINE_float('lstm_learning_rate', 1.0, 'lstm_learning_rate')
tf.app.flags.DEFINE_integer('lstm_nb_epoch', 30, 'lstm_nb_epoch')
tf.app.flags.DEFINE_string('lstm_checkpoints_dir', './checkpoints_var_lstm', 'checkpoints save path.')
tf.app.flags.DEFINE_string('lstm_model_prefix', 'LSTM_model', 'model save filename.')

FLAGS = tf.app.flags.FLAGS




def get_batch_words2id_data(lstm_batch_size):
    
    with open('./data/标点集.txt','r',encoding = 'utf-8') as f:
        characters = f.readlines()
    character_list = [i.split('\n')[0] for i in characters]
    character_dict = {i:0 for i in character_list}
    
    a = pd.read_excel('./data/neg.xls',header=None)
    a['sign'] = 0
    b = pd.read_excel('./data/pos.xls',header=None)
    b['sign'] = 1
    c = pd.concat([a,b])
    d = c.iloc[np.random.permutation(len(c))] 
    d['tags'] = list(map(lambda x:[0,1] if x == 1 else [1,0],d['sign']))
    d['words'] = d[0].apply(lambda i:list(jieba.cut(i)))    
    d['non_stop_words'] = d['words'].apply(lambda x:[k for k in x if k not in character_dict])
    
    allwords = []
    for i in d['non_stop_words']:
        allwords.extend(i)
    dictionary, words_tuple = build_dataset(allwords)
    d['words_id'] = d['non_stop_words'].apply(lambda x:[dictionary.get(j,0) for j in x])
    d['length'] = d['words_id'].apply(len)
    
    p = int(len(d)*2/3)
    for_train = d.iloc[:p]
    for_test = d.iloc[p:]
    
    train_test = (for_train, for_test)
    train_test_batches = []
    for data in train_test:
        sentences = data.sort_values('length')
    
        batch_sentences = []
        batch_tags = []
        batch_num = len(sentences)//lstm_batch_size
        for i in range(batch_num):
            start = i*lstm_batch_size
            end = (i+1)*lstm_batch_size
            batch_data = sentences['words_id'].iloc[start:end]
            batch_tag = np.array(sentences['tags'].iloc[start:end].tolist())
            length = max(map(len,batch_data))
        
            normal_batch_data = np.full((lstm_batch_size, length + 1), dictionary[' '], np.int32)
            for j in range(lstm_batch_size):
                end = len(batch_data.iloc[j])
                normal_batch_data[j,:end] = batch_data.iloc[j]
            batch_sentences.append(normal_batch_data)
            batch_tags.append(batch_tag)
        train_test_batches.append((batch_sentences, batch_tags))
        
             
    return train_test_batches[0][0], train_test_batches[0][1], train_test_batches[1][0], train_test_batches[1][1], dictionary, words_tuple



def lstm_train(update = False):

    train_x_batches, train_y_batches, test_x_batches, test_y_batches, word2id_dictionary, words_tuple = get_batch_words2id_data(FLAGS.lstm_batch_size)
    print(len(train_x_batches))    
    #用pickle保存word2id字典，为了后续预测数据时转换使用
    output = open('./dict/word2id_dictionary.pkl', 'wb')
    pickle.dump(word2id_dictionary, output)
    output.close()
    
    output1 = open('./dict/words_tuple.pkl', 'wb')
    pickle.dump(words_tuple, output1)
    output1.close()
            
    with tf.Graph().as_default():
        #设置整个graph的初始化方式
        initializer = tf.random_uniform_initializer(-FLAGS.lstm_init_scale, FLAGS.lstm_init_scale)

        x_train = tf.placeholder(tf.int32, shape=[FLAGS.lstm_batch_size, None])
        y_train = tf.placeholder(tf.float32, shape=[FLAGS.lstm_batch_size, 2])                
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = LSTMs_Model(len(words_tuple), FLAGS.lstm_cell_size, FLAGS.lstm_num_layers, FLAGS.lstm_keep_prob, x_train, y_train)
        
        x_test = tf.placeholder(tf.int32, shape=[FLAGS.lstm_batch_size, None])        
        with tf.variable_scope("Model", reuse=True, initializer=initializer):            
            m_test = LSTMs_Model(len(words_tuple), FLAGS.lstm_cell_size, FLAGS.lstm_num_layers, FLAGS.lstm_keep_prob, x_test)
            y_pre_tf = tf.argmax(m_test.predict,1)
            
        if update:
            ckpt = tf.train.get_checkpoint_state(FLAGS.lstm_checkpoints_dir)
        
        saver = tf.train.Saver()
        with tf.Session() as session:
            session.run(tf.global_variables_initializer()) 
            #再训练更新参数则还原之前的模型，继续训练，当然训练数据要变为新的数据
            if update:
                saver.restore(session, ckpt.model_checkpoint_path)

            for j in range(FLAGS.lstm_nb_epoch):
#            for j in range(1):
                S = 0
                n = 0 
                for i in range(len(train_x_batches)):
                    lstm_lr_decay_new = max(FLAGS.lstm_lr_decay ** max(i-1, 0.0)*0.01, 0.001)
                    session.run(m.lr_update, feed_dict = {m.new_lr: FLAGS.lstm_learning_rate * lstm_lr_decay_new})
                    _, cost = session.run([m.train_op, m.cost], feed_dict = {x_train: train_x_batches[i], y_train: train_y_batches[i]})
                    S += cost
                    n += 1                    
                    if i%100 == 0 and i > 0:
                        s = np.array([])
                        for k in range(len(test_x_batches)):
                            y_pre = session.run(y_pre_tf,feed_dict = {x_test: test_x_batches[k]})
                            y_tag = test_y_batches[k].argmax(axis = 1)       
                            a = (y_pre == y_tag)
                            s = np.concatenate((s,a))
                        accuracy = np.mean(s)
                        print("Epoch: %d batch_num: %d loss: %.3f, accuracy = %s" % (j, i, S/n, accuracy))
                        S = 0
                        n = 0
                        

            save_path = saver.save(session, FLAGS.lstm_checkpoints_dir + '/'+ FLAGS.lstm_model_prefix)
            print("Model saved in file: ", save_path)




    