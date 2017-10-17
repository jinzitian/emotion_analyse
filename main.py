# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:47:19 2017

@author: Jinzitian
"""

import sys
import numpy as np
import pandas as pd

from train.lstm_train import lstm_train
from predict.lstm_predict import lstm_predict
from train.cnn_lstm_train import cnn_lstm_train
from predict.cnn_lstm_predict import cnn_lstm_predict

def get_data():
    data1 = pd.read_excel(r'./data/pos.xls',header =None)
    data2 = pd.read_excel(r'./data/neg.xls',header =None)
    data = pd.concat([data1,data2],axis = 0)
    data.columns = ['user_content']
    return data
    
    
def main(args):
    
    if args[1] == 'train':
        if args[2] == 'lstm':
            lstm_train()
        elif args[2] == 'cnn_lstm':
            cnn_lstm_train()
        else:
            print('please try: python main.py train lstm/cnn_lstm')
			
    elif args[1] == 'update':
        if args[2] == 'lstm':
            lstm_train(True)
        elif args[2] == 'cnn_lstm':
            cnn_lstm_train(True)
        else:
            print('please try: python main.py update lstm/cnn_lstm')
			
    elif args[1] == 'example':
        data = get_data()
        sentence = [data['user_content'].iloc[np.random.randint(len(data))]]
        if args[2] == 'lstm':
            try:
                for s, e in lstm_predict(sentence):
                    print(s, '[' + e + ']')
            except Exception as e:
                print('you need train first')
                print('please try: python main.py train lstm')
        elif args[2] == 'cnn_lstm':
            try:
                for s, e in cnn_lstm_predict(sentence):
                   print(s, '[' + e + ']')
            except Exception as e:
                print('you need train first')
                print('please try: python main.py train cnn_lstm')
        else:
            print('please try: python main.py example lstm/cnn_lstm')
            
    elif args[1] == 'sentence':
        if args[2] == 'lstm':
            sentence = args[3:]
            try:
                for s,e in lstm_predict(sentence):
                    print(s, '[' + e + ']')
            except Exception as e:
                print('you need train first')
                print('please try: python main.py train lstm')				
        elif args[2] == 'cnn_lstm':
            sentence = args[3:]
            try:
                for s,e in cnn_lstm_predict(sentence):
                    print(s, '[' + e + ']')
            except Exception as e:
                print('you need train first')
                print('please try: python main.py train cnn_lstm')
        else:
            print('please try: python main.py sentence lstm/cnn_lstm instance1 instance2 ... instanceN')
		
    else:
        print('you can try like these:')
        print('  1、python main.py train lstm/cnn_lstm')
        print('  2、python main.py update lstm/cnn_lstm')
        print('  3、python main.py example lstm/cnn_lstm')
        print('  4、python main.py sentence lstm/cnn_lstm instance1 instance2 ... instanceN')
            
if __name__ == '__main__':
    
    main(sys.argv)
    


    
    
    
    