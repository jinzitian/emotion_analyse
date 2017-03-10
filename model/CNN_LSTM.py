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
        
        #构建LSTM循环单元
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(cell_size, forget_bias=0.0, state_is_tuple=True)
        #lstm单元的dropout算法，这里选择的是output_drop
        if self.keep_prob < 1:
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=keep_prob)
        #构建多层RNN网络
        cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * num_layers, state_is_tuple=True)
        self.cell = cell
        
        #设置各batch_size初始化的状态c、h为全零
        initial_state = self.cell.zero_state(batch_size, tf.float32)        
        
        self.vector_size = cell_size * 2
        #随机初始化embeding全连接层（即词向量神经元的前一层）
#        with tf.device("/cpu:0"):
#        获取变量时直接可以初始化：embedding = tf.get_variable('embedding', initializer = tf.random_uniform([1000, 128], -1.0, 1.0))
        embedding = tf.get_variable("embedding", shape = [vocabulary_size, self.vector_size], dtype=tf.float32)
#        embedding = tf.get_variable("embedding", [vocabulary_size, cell_size * 3], dtype=tf.float32)        
        self.embedding = embedding
        #本质上是one-hot输入向量通过embedding全连接层转换为词向量神经元的过程
        #inputs本质是神经网络的第一层全连接层
        inputs = tf.nn.embedding_lookup(self.embedding, input_data)

        #做第一层卷积，并通过relu输出，shape对应[卷积变换_height, 卷积变换_width, in_channels, out_channels]:[5, 5, 1, 32]
        #b_conv1中的 1 对应的是W_conv1中的[5,5,1,1]最后一个 1，也就是cnn中的通道数
        self.conv_height = 5
        self.conv_width = 5
        W_conv1 = tf.get_variable("W_conv1", [self.conv_height, self.conv_width,1,1], dtype=tf.float32)
        b_conv1 = tf.get_variable("b_conv1", [1], dtype=tf.float32)
        #h_conv1的shape为[batch, 卷积后的_height, 卷积后的_width, out_channels]:[batch, 24, 24, 32]
        #strides为窗口的步长，4个维度与输入对应，也就是[batch的步长, in_height的步长, in_width的步长, in_channels的步长]
        x_image = tf.reshape(inputs, [-1, num_steps, self.vector_size, 1])
        #卷积后的shape为 [batch, height, width, channel_num]，其中channel_num = 1
        inputs = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='VALID') + b_conv1)        
        #计算卷积后的高和宽
        num_steps = num_steps - self.conv_height + 1
        self.input_vector_size = self.vector_size - self.conv_width + 1
        #还原成LSTM接受的数据形式
        inputs = tf.reshape(inputs, [-1, num_steps, self.input_vector_size])          

        #词向量神经元作为输入的dropout优化
        if self.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, self.keep_prob)
        #高级接口的outputs的返回结构为[batch_size, numsteps, cell_size]
        outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state)
        #outputs是二维数组的列表，每一数组代表一个时刻所有batch_size的output
        #此处output的size为batch_size * cell_size
        output = outputs[:,-1,:]
        #tf0.12版本的命令
#        output = tf.concat_v2([outputs[-1]], 1)
        
        #情感分析的输出为2类
        if self.keep_prob < 1:
            output = tf.nn.dropout(output, self.keep_prob)
        V = tf.get_variable("V", [self.cell_size, 2], dtype=tf.float32)
        V_b = tf.get_variable("V_b", [2], dtype=tf.float32)
        
        
        #定义softmax分类输出
        softmax_output = tf.nn.softmax(tf.matmul(output, V) + V_b)
        self.predict = softmax_output
        
        if targets is not None:
            #用内置函数定义交叉熵，得到每条数据的损失列表
#            loss = tf.nn.softmax_cross_entropy_with_logits(labels=targets, logits=tf.matmul(output, V) + V_b)
#            average_loss = tf.reduce_mean(loss)
        
            #手动定义交叉熵定义损失函数
            
            loss = -targets*tf.log(softmax_output)
            
            average_loss = tf.reduce_mean(tf.reduce_sum(loss, axis = 1))
        
        
            #记录平均损失值
            self.cost = average_loss

            #损失函数优化算法
            self.lr = tf.Variable(0.0, trainable=False)
            optimizer = tf.train.AdamOptimizer(self.lr)
            self.train_op = optimizer.minimize(average_loss)
        
            #shape = []代表Tensor标量
            self.new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
            self.lr_update = tf.assign(self.lr, self.new_lr)        
        
        
        


if __name__ == '__main__':
    
    print('hello')
    


            

    

