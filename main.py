# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:47:19 2017

@author: Jinzitian
"""

import sys
import pandas as pd

from train.train import train
from predict.predict import predict

def get_data():
    data1 = pd.read_excel(r'./data/pos.xls',header =None)
    data2 = pd.read_excel(r'./data/neg.xls',header =None)
    data = pd.concat([data1,data2],axis = 0)
    data.columns = ['user_content']
    return data
    
    
def main(args):
    if len(args) != 2:
        print('please like this :python main.py train/example/sentence')
    else:
        if args[1] == 'train':
            train()
        elif args[1] == 'example':
            data = get_data()
            sentence = data['user_content'].iloc[1]
            print(sentence, '   ', predict(sentence))
        else:
            sentence = args[1]
            print(sentence, '   ', predict(sentence))
            
            
if __name__ == '__main__':
    
    main(sys.argv)