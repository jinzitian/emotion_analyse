# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 10:34:29 2017

@author: Administrator
"""

import collections


#将语料库转换为int型的id
def build_dataset(all_words):
    
    count_dict = collections.Counter(all_words)
    count_pairs = sorted(count_dict.items(), key = lambda x: -x[1])
    words, _ = zip(*count_pairs)
    words = ('other',) + words
    #这样构建的words的id和单词的位子刚好匹配，也就可以通过id来找到单词
    word2id = dict(zip(words,range(len(words))))
    
    return word2id, words
    