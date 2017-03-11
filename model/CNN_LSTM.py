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
            batch_size, num_steps = input_data.shape.as_list()
            print('train')
            print(batch_size, num_steps)
        elif isinstance(input_data, np.ndarray):
            batch_size, num_steps = input_data.shape
            print('predict')
            print(batch_size, num_steps)
        else:
            print('error')
            return
            
        self.vocabulary_size = vocabulary_size
        self.cell_size = cell_size
        self.num_layers = num_layers
        self.keep_prob = keep_prob
        
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(cell_size, forget_bias=0.0, state_is_tuple=True)
        if self.keep_prob < 1:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)
        self.cell = cell
        initial_state = self.cell.zero_state(batch_size, tf.float32)        
        
        self.vector_size = cell_size * 2
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", shape = [vocabulary_size, self.vector_size], dtype=tf.float32)     
            self.embedding = embedding
            inputs = tf.nn.embedding_lookup(self.embedding, input_data)

        self.conv_height = 5
        self.conv_width = 5
        W_conv1 = tf.get_variable("W_conv1", [self.conv_height, self.conv_width,1,1], dtype=tf.float32)
        b_conv1 = tf.get_variable("b_conv1", [1], dtype=tf.float32)
        x_image = tf.reshape(inputs, [-1, num_steps, self.vector_size, 1])
        inputs = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='VALID') + b_conv1)        
        num_steps = num_steps - self.conv_height + 1
        self.input_vector_size = self.vector_size - self.conv_width + 1
        inputs = tf.reshape(inputs, [-1, num_steps, self.input_vector_size])          

        if self.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, self.keep_prob)
        outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
        output = outputs[:,-1,:]

        
        if self.keep_prob < 1:
            output = tf.nn.dropout(output, self.keep_prob)
        V = tf.get_variable("V", [self.cell_size, 2], dtype=tf.float32)
        V_b = tf.get_variable("V_b", [2], dtype=tf.float32)
        softmax_output = tf.nn.softmax(tf.matmul(output, V) + V_b)
        self.predict = softmax_output
        
        if targets is not None:
            #用内置函数定义交叉熵，得到每条数据的损失列表
#            loss = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=tf.matmul(output, V) + V_b)
#            average_loss = tf.reduce_mean(loss)
            #手动定义交叉熵定义损失函数
            loss = -targets*tf.log(softmax_output)
            average_loss = tf.reduce_mean(tf.reduce_sum(loss, axis = 1))

            self.cost = average_loss

            self.lr = tf.Variable(0.0, trainable=False)
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(average_loss)
    
            self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
            self.lr_update = tf.assign(self.lr, self.new_lr)        
        
        

if __name__ == '__main__':
    
    print('hello')
    


            

    

