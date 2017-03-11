# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:47:19 2017

@author: Jinzitian
"""

import sys
import pandas as pd
import numpy as np

from train.train import train
from predict.predict import predict

def get_data():
    data1 = pd.read_excel(r'./data/pos.xls',header =None)
    data2 = pd.read_excel(r'./data/neg.xls',header =None)
    data = pd.concat([data1,data2],axis = 0)
    data.columns = ['user_content']
    return data
    
    
def main(args):
    
    if args[1] == 'train':
        train()
    elif args[1] == 'example':
        data = get_data()
        sentence = [data['user_content'].iloc[np.random.randint(len(data))]]
        try:
            for s, e in predict(sentence):
                print(s, '[' + e + ']')
        except Exception as e:
            print('you need train first')
            
    elif args[1] == 'sentence':
        sentence = args[2:]
        try:
            for s,e in predict(sentence):
                print(s, '[' + e + ']')
        except Exception as e:
            print('you need train first')
    else:
        print('you can try like these:')
        print('  1、python main.py train')
        print('  2、python main.py example')
        print('  3、python main.py sentence instance1 instance2 ... instanceN')
            
if __name__ == '__main__':
    
    main(sys.argv)
    
    
    
    
    