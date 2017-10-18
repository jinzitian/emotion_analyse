# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 13:22:59 2017

@author: Jinzitian
"""



import numpy as np
import tensorflow as tf



class LSTMs_Model(object):

    def __init__(self, vocabulary_size, cell_size, num_layers, keep_prob, input_data, targets = None):
        
        if isinstance(input_data, tf.Tensor):            
            batch_size = input_data.shape[0].value
            if targets is not None:
                print('train, batch_size is %s'%batch_size)
            else:
                print('predict, batch_size is %s'%batch_size)
        elif isinstance(input_data, np.ndarray):
            batch_size = input_data.shape[0]
            print('predict, batch_size is %s'%batch_size)
        else:
            print('error')
            return
            
        self.vocabulary_size = vocabulary_size
        self.cell_size = cell_size
        self.num_layers = num_layers
        self.keep_prob = keep_prob
        
        def single_cell():
            return tf.contrib.rnn.BasicLSTMCell(cell_size, forget_bias=0.0, state_is_tuple=True)
        
        lstm_cell = single_cell
        
        if self.keep_prob < 1 and targets is not None:
            def dropout_single_cell():
                return tf.contrib.rnn.DropoutWrapper(single_cell(), output_keep_prob=keep_prob)
            lstm_cell = dropout_single_cell
            
        #创建多层lstm需要注意，每层的lstm_cell都是独立的对象，需要单独创建
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell() for i in range(num_layers)], state_is_tuple=True)
        self.cell = cell
        initial_state = self.cell.zero_state(batch_size, tf.float32)        
        
        self.vector_size = cell_size * 2
        
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", shape = [vocabulary_size, self.vector_size], dtype=tf.float32)     
            self.embedding = embedding
            inputs = tf.nn.embedding_lookup(self.embedding, input_data)
        
        #词向量先转换为lstm的cell长度，输入lstm
        #vec_lstm_weight = tf.get_variable("vec_lstm_weight", shape = [self.vector_size, self.cell_size], dtype=tf.float32) 
        #inputs = tf.reshape(tf.matmul(tf.reshape(inputs,[-1,self.vector_size]), vec_lstm_weight),[batch_size,-1,self.cell_size])

#        self.conv_height = 5
#        self.conv_width = 5
#        W_conv1 = tf.get_variable("W_conv1", [self.conv_height, self.conv_width,1,1], dtype=tf.float32)
#        b_conv1 = tf.get_variable("b_conv1", [1], dtype=tf.float32)
#        x_image = tf.reshape(inputs, [batch_size, -1, self.vector_size, 1])
#        inputs = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='VALID') + b_conv1)        
#        self.input_vector_size = self.vector_size - self.conv_width + 1
#        inputs = tf.reshape(inputs, [batch_size, -1, self.input_vector_size])  
        
        if self.keep_prob < 1 and targets is not None:
            inputs = tf.nn.dropout(inputs, self.keep_prob)
    
        outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
        output = outputs[:,-1,:]
        
        
        if self.keep_prob < 1 and targets is not None:
            output = tf.nn.dropout(output, self.keep_prob)
        V = tf.get_variable("V", [self.cell_size, 2], dtype=tf.float32)
        V_b = tf.get_variable("V_b", [2], dtype=tf.float32)
        softmax_output = tf.nn.softmax(tf.matmul(output, V) + V_b)
        self.predict = softmax_output
        
        if targets is not None:
            #用内置函数定义交叉熵，得到每条数据的损失列表
            loss = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=tf.matmul(output, V) + V_b)
            average_loss = tf.reduce_mean(loss)
            #手动定义交叉熵定义损失函数,如果为log(0)会出现nan
#            loss = -targets*tf.log(softmax_output)
#            average_loss = tf.reduce_mean(tf.reduce_sum(loss, axis = 1))

            self.cost = average_loss

            self.lr = tf.Variable(0.0, trainable=False)
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(average_loss)
    
            self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
            self.lr_update = tf.assign(self.lr, self.new_lr)        
        
        

if __name__ == '__main__':
    
    print('hello')
    


            

    

