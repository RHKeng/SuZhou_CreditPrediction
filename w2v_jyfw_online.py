# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import time
import datetime
import gc
import jieba
import functools
from gensim.models import Word2Vec

#定义jieba分词函数
def jieba_sentences(sentence):
    seg_list = jieba.cut(sentence)
    seg_list = list(seg_list)
    return seg_list

##-----------------------------------------------------------------------------
if __name__=='__main__':
    now = datetime.datetime.now()
    now = now.strftime('%m-%d-%H-%M')
    print(now)

    chusai_JGSLDJXX_train_df = pd.read_csv('../data/chusai/train/机构设立（变更）登记信息.csv')
    fusai_JGSLDJXX_train_df = pd.read_csv('../data/fusai/train/机构设立（变更）登记信息.csv')
    JGSLDJXX_test_df = pd.read_csv('../data/fusai/test/机构设立（变更）登记信息.csv')
    total_df = pd.concat([chusai_JGSLDJXX_train_df, fusai_JGSLDJXX_train_df, JGSLDJXX_test_df])

    sentence_list = total_df[total_df['经营范围'].notnull()]['经营范围'].tolist()
    sentence_list = [jieba_sentences(x) for x in sentence_list]

    my_model = Word2Vec(sentence_list, size=10, window=5, sg=1, hs=1, min_count=2, workers=1, seed=0)
    my_model.save('../temp/jyfw_w2v_online.model')

    now = datetime.datetime.now()
    now = now.strftime('%m-%d-%H-%M')
    print(now)
