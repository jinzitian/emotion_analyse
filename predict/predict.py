# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 11:44:40 2016

@author: Jinzitian
"""

import pickle
import jieba
import numpy as np
import tensorflow as tf

from model.CNN_LSTM import LSTMs_Model
from train.train import FLAGS

def predict(sentence):
    
    picklefile = open(r'./dict/word2id_dictionary.pkl', 'rb')
    word2id_dictionary = pickle.load(picklefile)
    picklefile.close()    
    
    picklefile = open(r'./dict/words_tuple.pkl', 'rb')
    words_tuple = pickle.load(picklefile)
    picklefile.close()    
    
    jieba.add_word('难喝')
    words_list = jieba.cut(sentence)
    with open(r'./data/标点集.txt','r',encoding = 'utf-8') as f:
        signs = f.readlines()
    signs_list = [i.split('\n')[0] for i in signs]
    signs_dict = {i:0 for i in signs_list}
    
    words_list_without_signs = [i for i in words_list if i not in signs_dict]
    id_list = [word2id_dictionary.get(i,0) for i in words_list_without_signs]
    predict_id_list = id_list + [0]*(FLAGS.num_steps-len(id_list)) if len(id_list) < FLAGS.num_steps else id_list[: FLAGS.num_steps]
    predict_id_list = np.array([predict_id_list])    
    
    with tf.Graph().as_default():
        #设置整个graph的初始化方式
        input_data = predict_id_list
        targets = None
        initializer = tf.random_uniform_initializer(-FLAGS.init_scale, FLAGS.init_scale)
        with tf.variable_scope("Model", reuse=None, initializer=initializer):
            m = LSTMs_Model(len(words_tuple), FLAGS.cell_size, FLAGS.num_layers, FLAGS.keep_prob, input_data, targets)
            #整句话判断的情况  
            y_pre_tf = tf.argmax(m.predict,1)
            
            ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoints_dir)
            saver = tf.train.Saver()
        
            with tf.Session() as session:
                #初始化所有参数
                session.run(tf.global_variables_initializer())
                #恢复模型的参数覆盖初始化中名称相同的全部参数，如果初始化时有新增的参数，新增参数不会受到影响
                saver.restore(session, ckpt.model_checkpoint_path)
         
                #整句话判断的情况
                y_pre = session.run(y_pre_tf)              
                emotion = '正面情感' if y_pre == 1 else '负面情感'
                
                return emotion






    

    