#!/usr/bin/env python
# -*-coding:utf-8-*-

'''

'''

import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import csv
import time
import math
import functools
from datetime import *
import re
from gensim.models import Word2Vec
import jieba

#开始统计跟出资比例相关的统计特征
def get_QY_czbl_statistic_fea(df, QY_df):
    temp_pivot_table = pd.pivot_table(QY_df, index='qymc', values='czbl', aggfunc={np.max, np.mean, np.min, np.std})
    temp_pivot_table.reset_index(inplace=True)
    temp_pivot_table.rename(columns={'amax':'czbl_max', 'amin':'czbl_min', 'mean':'czbl_mean', 'std':'czbl_std'}, inplace=True)
    df = pd.merge(df, temp_pivot_table, on='qymc', how='left')
    return df

#定义获取两列时间差函数
def get_deltaDay(df, col1, col2):
    col1 = df[col1]
    col2 = df[col2]
    if (col1 is np.nan) | (col2 is np.nan) | (col1 == -1) | (col2 == -1):
        return np.nan
    else:
        return (col2 - col1).days

#开始处理跟企业成立日期，核准日期相关的特征
def get_QY_clhzrq_feature(df):
    df['clrq'] = df['clrq'].map(lambda x : np.nan if x is np.nan else pd.to_datetime(str(x)[:10]))
    df['hzrq'] = df['hzrq'].map(lambda x : np.nan if x is np.nan else pd.to_datetime(str(x)[:10]))
    df['clrq_year'] = df['clrq'].map(lambda x : np.nan if x is np.nan else x.year)
    df['clrq_month'] = df['clrq'].map(lambda x : np.nan if x is np.nan else x.month)
    df['clrq_day'] = df['clrq'].map(lambda x : np.nan if x is np.nan else x.day)
    df['hzrq_year'] = df['hzrq'].map(lambda x : np.nan if x is np.nan else x.year)
    df['hzrq_month'] = df['hzrq'].map(lambda x : np.nan if x is np.nan else x.month)
    df['hzrq_day'] = df['hzrq'].map(lambda x : np.nan if x is np.nan else x.day)
    f = functools.partial(get_deltaDay, col1='clrq', col2='hzrq')
    df['clhz_deltaDay'] = df.apply(f, axis=1)
    return df

#开始统计投资人个数特征
def get_QY_tzrgs_feature(df, QY_xinxi_df):
    QY_xinxi_df_cpoy = QY_xinxi_df.copy()
    QY_xinxi_df_cpoy.drop_duplicates(['qymc', 'tzr'], inplace=True)
    temp_pivot_table = pd.pivot_table(QY_xinxi_df_cpoy, index='qymc', values='tzr', aggfunc=len)
    temp_pivot_table.reset_index(inplace=True)
    temp_pivot_table.rename(columns={'tzr':'QY_tzrgs'}, inplace=True)
    df = pd.merge(df, temp_pivot_table, on='qymc', how='left')
    return df

#开始统计职务种类数特征
def get_QY_zwzls_feature(df, QY_xinxi_df):
    QY_xinxi_df_cpoy = QY_xinxi_df.copy()
    QY_xinxi_df_cpoy.drop_duplicates(['qymc', 'zw'], inplace=True)
    temp_pivot_table = pd.pivot_table(QY_xinxi_df_cpoy, index='qymc', values='zw', aggfunc=len)
    temp_pivot_table.reset_index(inplace=True)
    temp_pivot_table.rename(columns={'zw':'QY_zwzls'}, inplace=True)
    df = pd.merge(df, temp_pivot_table, on='qymc', how='left')
    return df

#开始统计法定代表人个数，首席代表个数
def get_QY_dbgs_feature(df, QY_xinxi_df):
    for fea in ['fddbrbz', 'sxdbbz']:
        temp = QY_xinxi_df[QY_xinxi_df[fea] == '是']
        temp_pivot_table = pd.pivot_table(temp, index='qymc', values=fea, aggfunc=len)
        temp_pivot_table.reset_index(inplace=True)
        temp_pivot_table.rename(columns={fea:'QY_' + fea + '_gs'}, inplace=True)
        df = pd.merge(df, temp_pivot_table, on='qymc', how='left')
    return df

#初步统计企业信息记录数
def get_QYJL_number_feature(df, QY_xinxi_df):
    temp_pivot_table = pd.pivot_table(QY_xinxi_df, index='qymc', values='clrq', aggfunc=len)
    temp_pivot_table.reset_index(inplace=True)
    temp_pivot_table.rename(columns={'clrq':'QYJL_number'}, inplace=True)
    df = pd.merge(df, temp_pivot_table, on='qymc', how='left')
    df['QYJL_number'] = df['QYJL_number'].fillna(0)
    return df

#统计跟分支成立死亡相关的特征
def get_FZ_clsw_feature(df):
    df['fzclsj'] = df['fzclsj'].map(lambda x : np.nan if ((x is np.nan) | (x == -1)) else datetime(int(x.split('/')[0]), int(x.split('/')[1]), int(x.split('/')[2])))
    df['fzswsj'] = df['fzswsj'].map(lambda x : np.nan if ((x is np.nan) | (x == -1)) else datetime(int(x.split('/')[0]), int(x.split('/')[1]), int(x.split('/')[2])))
    df['is_FZ_die'] = df['fzswsj'].map(lambda x : 1 if ((x != np.nan) & (x != -1)) else 0)
    f = functools.partial(get_deltaDay, col1='fzclsj', col2='fzswsj')
    df['fzsw_deltaDay'] = df.apply(f, axis=1)
    return df

#定义jieba分词函数
def jieba_sentences(sentence):
    seg_list = jieba.cut(sentence)
    seg_list = list(seg_list)
    return seg_list

def get_dir_w2v_array(word_list, word_wv, num_features):
    word_vectors = np.zeros((len(word_list), num_features))
    for i in range(len(word_list)):
        if str(word_list[i]) in word_wv.vocab.keys():
            word_vectors[i][:] = word_wv[str(word_list[i])]
    mean_array = np.mean(word_vectors, axis=0)
    return mean_array

def get_jyfw_w2v_feature(df, dir_num_features):
    for i in range(dir_num_features):
        df['jyfw_array_' + str(i) + '_fea'] = df['jyfw_array'].map(lambda x: np.nan if len(x) < (i+1) else x[i])
    return df

#初步统计法人行政许可注销信息
def get_FRXZXKZX_number_feature(df, FRXZXKZX_df):
    temp_pivot_table = pd.pivot_table(FRXZXKZX_df, index='企业名称', values='提供日期', aggfunc=len)
    temp_pivot_table.reset_index(inplace=True)
    temp_pivot_table.rename(columns={'提供日期':'FRXZXKZX_number', '企业名称':'qymc'}, inplace=True)
    df = pd.merge(df, temp_pivot_table, on='qymc', how='left')
    df['FRXZXKZX_number'] = df['FRXZXKZX_number'].fillna(0)
    return df

#初步统计企业表彰荣誉信息
def get_QYBZRYXX_number_feature(df, QYBZRYXX_df):
    temp_pivot_table = pd.pivot_table(QYBZRYXX_df, index='企业名称', values='创建时间', aggfunc=len)
    temp_pivot_table.reset_index(inplace=True)
    temp_pivot_table.rename(columns={'创建时间':'QYBZRYXX_number', '企业名称':'qymc'}, inplace=True)
    df = pd.merge(df, temp_pivot_table, on='qymc', how='left')
    df['QYBZRYXX_number'] = df['QYBZRYXX_number'].fillna(0)
    return df

#初步统计企业非正常户认定
def get_QYFZCHRD_number_feature(df, QYFZCHRD_df):
    temp_pivot_table = pd.pivot_table(QYFZCHRD_df, index='qymc', values='cjsj', aggfunc=len)
    temp_pivot_table.reset_index(inplace=True)
    temp_pivot_table.rename(columns={'cjsj':'QYFZCHRD_number'}, inplace=True)
    df = pd.merge(df, temp_pivot_table, on='qymc', how='left')
    df['QYFZCHRD_number'] = df['QYFZCHRD_number'].fillna(0)
    return df

#将企业非正常用户的一些字段merge到训练集中
def get_QYFZCHRD_basic_feature(df, QYFZCHRD_df):
    QYFZCHRD_df_copy = QYFZCHRD_df.copy()
    QYFZCHRD_df_copy.drop_duplicates('qymc', inplace=True)
    QYFZCHRD_df_copy.rename(columns={'rwbh':'QYFZCHRD_rwbh', 'zcdz':'QYFZCHRD_zcdz', 'xxtgbmmc':'QYFZCHRD_xxtgbmmc', 'swglm':'QYFZCHRD_swglm', 'gljg':'QYFZCHRD_gljg', 'djzclx':'QYFZCHRD_djzclx', 'yynx':'QYFZCHRD_yynx'}, inplace=True)
    df = pd.merge(df, QYFZCHRD_df_copy[['qymc', 'QYFZCHRD_rwbh', 'QYFZCHRD_zcdz', 'QYFZCHRD_xxtgbmmc', 'QYFZCHRD_swglm', 'QYFZCHRD_gljg', 'QYFZCHRD_djzclx', 'QYFZCHRD_yynx']], on='qymc', how='left')
    return df

#初步统计企业税务登记信息
def get_QYSWDJXX_number_feature(df, QYSWDJXX_df):
    temp_pivot_table = pd.pivot_table(QYSWDJXX_df, index='qymc', values='cjsj', aggfunc=len)
    temp_pivot_table.reset_index(inplace=True)
    temp_pivot_table.rename(columns={'cjsj':'QYSWDJXX_number'}, inplace=True)
    df = pd.merge(df, temp_pivot_table, on='qymc', how='left')
    df['QYSWDJXX_number'] = df['QYSWDJXX_number'].fillna(0)
    QYSWDJXX_df_copy = QYSWDJXX_df.copy()
    QYSWDJXX_df_copy.drop_duplicates('qymc', inplace=True)
    QYSWDJXX_df_copy.rename(columns={'fddbrzjmc':'QYSWDJXX_fddbrzjmc', 'shdw':'QYSWDJXX_shdw', 'qy':'QYSWDJXX_qy', 'shjg':'QYSWDJXX_shjg', 'djzclx':'QYSWDJXX_djzclx'}, inplace=True)
    df = pd.merge(df, QYSWDJXX_df_copy[['qymc', 'QYSWDJXX_fddbrzjmc', 'QYSWDJXX_shdw', 'QYSWDJXX_qy', 'QYSWDJXX_shjg', 'QYSWDJXX_djzclx']], on='qymc', how='left')
    return df

#初步统计双打办打击侵权假冒处罚案件信息
def get_SDBDJQQJMCF_number_feature(df, SDBDJQQJMCF_df):
    temp_pivot_table = pd.pivot_table(SDBDJQQJMCF_df, index='qymc', values='cjsj', aggfunc=len)
    temp_pivot_table.reset_index(inplace=True)
    temp_pivot_table.rename(columns={'cjsj':'SDBDJQQJMCF_number'}, inplace=True)
    df = pd.merge(df, temp_pivot_table, on='qymc', how='left')
    df['SDBDJQQJMCF_number'] = df['SDBDJQQJMCF_number'].fillna(0)
    return df

#初步统计双公示-法人行政许可信息
def get_SGSFRXZXKXX_number_feature(df, SGSFRXZXKXX_df):
    temp_pivot_table = pd.pivot_table(SGSFRXZXKXX_df, index='qymc', values='cjsj', aggfunc=len)
    temp_pivot_table.reset_index(inplace=True)
    temp_pivot_table.rename(columns={'cjsj':'SGSFRXZXKXX_number'}, inplace=True)
    df = pd.merge(df, temp_pivot_table, on='qymc', how='left')
    df['SGSFRXZXKXX_number'] = df['SGSFRXZXKXX_number'].fillna(0)
    return df

#开始统计跟法人行政许可相关的信息
def get_FRXZXK_last_statistic_feature(df, SGSFRXZXKXX_df):
    SGSFRXZXKXX_df_copy = SGSFRXZXKXX_df.copy()
    SGSFRXZXKXX_df_copy.sort_values(by=['qymc', 'xkjzq'], ascending=False, inplace=True)
    SGSFRXZXKXX_df_copy.drop_duplicates('qymc', keep='first', inplace=True)
    SGSFRXZXKXX_df_copy.rename(columns={'xzxkbm':'SGSFRXZXKXX_xzxkbm', 'xkjdswh':'SGSFRXZXKXX_xkjdswh', 'xxtgbmmc':'SGSFRXZXKXX_xxtgbmmc', 'xknr':'SGSFRXZXKXX_xknr', 'xmmc':'SGSFRXZXKXX_xmmc', 'splb':'SGSFRXZXKXX_splb', 'dfbm':'SGSFRXZXKXX_dfbm', 'xkjg':'SGSFRXZXKXX_xkjg', 'xkjzq':'SGSFRXZXKXX_xkjzq', 'xysyfw':'SGSFRXZXKXX_xysyfw', 'sjzt2':'SGSFRXZXKXX_sjzt2', 'xkjdrq':'SGSFRXZXKXX_xkjdrq'}, inplace=True)
    df = pd.merge(df, SGSFRXZXKXX_df_copy[['qymc', 'SGSFRXZXKXX_xzxkbm', 'SGSFRXZXKXX_xkjdswh', 'SGSFRXZXKXX_xxtgbmmc', 'SGSFRXZXKXX_xknr', 'SGSFRXZXKXX_xmmc', 'SGSFRXZXKXX_splb', 'SGSFRXZXKXX_dfbm', 'SGSFRXZXKXX_xkjg', 'SGSFRXZXKXX_xkjzq', 'SGSFRXZXKXX_xkjdrq', 'SGSFRXZXKXX_xysyfw', 'SGSFRXZXKXX_sjzt2']], on='qymc', how='left')
    df['SGSFRXZXKXX_xkjzq'] = df['SGSFRXZXKXX_xkjzq'].map(lambda x : np.nan if x is np.nan else pd.to_datetime(x))
    df['SGSFRXZXKXX_xkjdrq'] = df['SGSFRXZXKXX_xkjdrq'].map(lambda x : np.nan if x is np.nan else pd.to_datetime(x))
    df['SGSFRXZXKXX_xkjzq_year'] = df['SGSFRXZXKXX_xkjzq'].map(lambda x : np.nan if x is np.nan else x.year)
    df['SGSFRXZXKXX_xkjzq_month'] = df['SGSFRXZXKXX_xkjzq'].map(lambda x : np.nan if x is np.nan else x.month)
    df['SGSFRXZXKXX_xkjzq_day'] = df['SGSFRXZXKXX_xkjzq'].map(lambda x : np.nan if x is np.nan else x.day)
    df['SGSFRXZXKXX_xkjdrq_year'] = df['SGSFRXZXKXX_xkjdrq'].map(lambda x : np.nan if x is np.nan else x.year)
    df['SGSFRXZXKXX_xkjdrq_month'] = df['SGSFRXZXKXX_xkjdrq'].map(lambda x : np.nan if x is np.nan else x.month)
    df['SGSFRXZXKXX_xkjdrq_day'] = df['SGSFRXZXKXX_xkjdrq'].map(lambda x : np.nan if x is np.nan else x.day)
    f1 = functools.partial(get_deltaDay, col1='SGSFRXZXKXX_xkjdrq', col2='SGSFRXZXKXX_xkjzq')
    df['SGSFRXZXKXX_xkjdjz_deltaDay'] = df.apply(f1, axis=1)
    f2 = functools.partial(get_deltaDay, col1='clrq', col2='SGSFRXZXKXX_xkjzq')
    df['clxkjz_deltaDay'] = df.apply(f2, axis=1)
    f3 = functools.partial(get_deltaDay, col1='hzrq', col2='SGSFRXZXKXX_xkjzq')
    df['hzxkjz_deltaDay'] = df.apply(f3, axis=1)
    f4 = functools.partial(get_deltaDay, col1='clrq', col2='SGSFRXZXKXX_xkjdrq')
    df['clxkjd_deltaDay'] = df.apply(f4, axis=1)
    f5 = functools.partial(get_deltaDay, col1='hzrq', col2='SGSFRXZXKXX_xkjdrq')
    df['hzxkjd_deltaDay'] = df.apply(f5, axis=1)
    return df

#初步统计许可资质年检信息
def get_XKZZNJXX_number_feature(df, XKZZNJXX_df):
    temp_pivot_table = pd.pivot_table(XKZZNJXX_df, index='qymc', values='cjsj', aggfunc=len)
    temp_pivot_table.reset_index(inplace=True)
    temp_pivot_table.rename(columns={'cjsj':'XKZZNJXX_number'}, inplace=True)
    df = pd.merge(df, temp_pivot_table, on='qymc', how='left')
    df['XKZZNJXX_number'] = df['XKZZNJXX_number'].fillna(0)
    return df

#开始统计跟许可资质年检信息
def get_XKZZNJXX_last_statistic_feature(df, XKZZNJXX_df):
    XKZZNJXX_df_copy = XKZZNJXX_df.copy()
    XKZZNJXX_df_copy.sort_values(by=['qymc', 'njrq'], ascending=False, inplace=True)
    XKZZNJXX_df_copy.drop_duplicates('qymc', keep='first', inplace=True)
    XKZZNJXX_df_copy.rename(columns={'xxtgbmmc':'XKZZNJXX_xxtgbmmc', 'njrq':'XKZZNJXX_njrq', 'njnd':'XKZZNJXX_njnd', 'njjgmc':'XKZZNJXX_njjgmc', 'njsxmc':'XKZZNJXX_njsxmc'}, inplace=True)
    df = pd.merge(df, XKZZNJXX_df_copy[['qymc', 'XKZZNJXX_xxtgbmmc', 'XKZZNJXX_njrq', 'XKZZNJXX_njnd', 'XKZZNJXX_njjgmc', 'XKZZNJXX_njsxmc']], on='qymc', how='left')
    df['XKZZNJXX_njrq'] = df['XKZZNJXX_njrq'].map(lambda x : np.nan if x is np.nan else pd.to_datetime(x))
    df['XKZZNJXX_njrq_year'] = df['XKZZNJXX_njrq'].map(lambda x : np.nan if x is np.nan else x.year)
    df['XKZZNJXX_njrq_month'] = df['XKZZNJXX_njrq'].map(lambda x : np.nan if x is np.nan else x.month)
    df['XKZZNJXX_njrq_day'] = df['XKZZNJXX_njrq'].map(lambda x : np.nan if x is np.nan else x.day)
    f1 = functools.partial(get_deltaDay, col1='hzrq', col2='XKZZNJXX_njrq')
    df['njhz_deltaDay'] = df.apply(f1, axis=1)
    f2 = functools.partial(get_deltaDay, col1='clrq', col2='XKZZNJXX_njrq')
    df['njcl_deltaDay'] = df.apply(f2, axis=1)
    return df

#初步统计招聘数据
def get_ZPSJ_number_feature(df, ZPSJ_df):
    temp_pivot_table = pd.pivot_table(ZPSJ_df, index='qymc', values='zprq', aggfunc=len)
    temp_pivot_table.reset_index(inplace=True)
    temp_pivot_table.rename(columns={'zprq':'ZPSJ_number'}, inplace=True)
    df = pd.merge(df, temp_pivot_table, on='qymc', how='left')
    df['ZPSJ_number'] = df['ZPSJ_number'].fillna(0)
    return df

#开始统计跟招聘信息种类相关的特征
def get_ZPSJ_kind_feature(df, ZPSJ_df):
    fea_list = ['zylb', 'gzjy', 'gzdd', 'zprs', 'zwyx', 'zdxl', 'zprq']
    for fea in fea_list:
        temp_df_pivot_table = pd.pivot_table(ZPSJ_df, index=['qymc', fea], values='wzmc', aggfunc=len)
        temp_df_pivot_table.reset_index(inplace=True)
        temp = pd.pivot_table(temp_df_pivot_table, index='qymc', values=fea, aggfunc=len)
        temp.reset_index(inplace=True)
        temp.rename(columns={fea:'ZPSJ_' + fea + '_kinds'}, inplace=True)
        df = pd.merge(df, temp, on='qymc', how='left')
    return df

#统计跟招聘人数相关的统计特征
def get_ZPSJ_zprs_statistic_feature(df, ZPSJ_df):
    mode = re.compile(r'\d+')
    ZPSJ_df['zprs'] = ZPSJ_df['zprs'].map(lambda x : np.nan if x is np.nan else (np.nan if len(mode.findall(str(x))) <= 0 else int(mode.findall(str(x))[0])))
    temp = ZPSJ_df[ZPSJ_df.zprs.notnull()]
    temp_pivot_table = pd.pivot_table(temp, index='qymc', values='zprs', aggfunc=[np.max, np.min, np.mean, np.std])
    temp_pivot_table.reset_index(inplace=True)
    temp_pivot_table.columns = ['qymc', 'ZPSJ_zprs_max', 'ZPSJ_zprs_min', 'ZPSJ_zprs_mean', 'ZPSJ_zprs_std']
    df = pd.merge(df, temp_pivot_table, on='qymc', how='left')
    return df

#初步统计资质登记（变更）信息
def get_ZZDJBGXX_number_feature(df, ZZDJBGXX_df):
    temp_pivot_table = pd.pivot_table(ZZDJBGXX_df, index='qymc', values='cjsj', aggfunc=len)
    temp_pivot_table.reset_index(inplace=True)
    temp_pivot_table.rename(columns={'cjsj':'ZZDJBGXX_number'}, inplace=True)
    df = pd.merge(df, temp_pivot_table, on='qymc', how='left')
    df['ZZDJBGXX_number'] = df['ZZDJBGXX_number'].fillna(0)
    return df

#开始统计跟资质登记（变更）信息
def get_ZZDJBGXX_last_statistic_feature(df, ZZDJBGXX_df):
    ZZDJBGXX_df_copy = ZZDJBGXX_df.copy()
    ZZDJBGXX_df_copy.sort_values(by=['qymc', 'zzjzq'], ascending=False, inplace=True)
    ZZDJBGXX_df_copy.drop_duplicates('qymc', keep='first', inplace=True)
    ZZDJBGXX_df_copy.rename(columns={'zzmc':'ZZDJBGXX_zzmc', 'zyfw':'ZZDJBGXX_zyfw', 'zzsxq':'ZZDJBGXX_zzsxq', 'zzjzq':'ZZDJBGXX_zzjzq', 'rdjgqc':'ZZDJBGXX_rdjgqc', 'zzzsbh':'ZZDJBGXX_zzzsbh'}, inplace=True)
    df = pd.merge(df, ZZDJBGXX_df_copy[['qymc', 'ZZDJBGXX_zzmc', 'ZZDJBGXX_zyfw', 'ZZDJBGXX_zzsxq', 'ZZDJBGXX_zzjzq', 'ZZDJBGXX_rdjgqc', 'ZZDJBGXX_zzzsbh']], on='qymc', how='left')
    df['ZZDJBGXX_zzsxq'] = df['ZZDJBGXX_zzsxq'].map(lambda x : np.nan if x is np.nan else pd.to_datetime(x))
    df['ZZDJBGXX_zzsxq_year'] = df['ZZDJBGXX_zzsxq'].map(lambda x : np.nan if x is np.nan else x.year)
    df['ZZDJBGXX_zzsxq_month'] = df['ZZDJBGXX_zzsxq'].map(lambda x : np.nan if x is np.nan else x.month)
    df['ZZDJBGXX_zzsxq_day'] = df['ZZDJBGXX_zzsxq'].map(lambda x : np.nan if x is np.nan else x.day)
    df['ZZDJBGXX_zzjzq'] = df['ZZDJBGXX_zzjzq'].map(lambda x : np.nan if x is np.nan else pd.to_datetime(x))
    df['ZZDJBGXX_zzjzq_year'] = df['ZZDJBGXX_zzjzq'].map(lambda x : np.nan if x is np.nan else x.year)
    df['ZZDJBGXX_zzjzq_month'] = df['ZZDJBGXX_zzjzq'].map(lambda x : np.nan if x is np.nan else x.month)
    df['ZZDJBGXX_zzjzq_day'] = df['ZZDJBGXX_zzjzq'].map(lambda x : np.nan if x is np.nan else x.day)
    f = functools.partial(get_deltaDay, col1='ZZDJBGXX_zzsxq', col2='ZZDJBGXX_zzjzq')
    df['ZZDJBGXX_jzsx_deltaDay'] = df.apply(f, axis=1)
    f1 = functools.partial(get_deltaDay, col1='clrq', col2='ZZDJBGXX_zzsxq')
    df['zzsxcl_deltaDay'] = df.apply(f1, axis=1)
    f2 = functools.partial(get_deltaDay, col1='clrq', col2='ZZDJBGXX_zzjzq')
    df['zzjzcl_deltaDay'] = df.apply(f2, axis=1)
    f3 = functools.partial(get_deltaDay, col1='hzrq', col2='ZZDJBGXX_zzsxq')
    df['zzsxhz_deltaDay'] = df.apply(f3, axis=1)
    f4 = functools.partial(get_deltaDay, col1='hzrq', col2='ZZDJBGXX_zzjzq')
    df['zzjzhz_deltaDay'] = df.apply(f4, axis=1)
    return df

def get_type_future_encoder(train_df, test_df, col_name):
    df_copy = pd.concat([train_df[[col_name]], test_df[[col_name]]])
    df_copy[col_name] = df_copy[col_name].astype(str)
    df_copy.sort_values(by = col_name, inplace = True)
    df_copy.drop_duplicates([col_name], inplace = True)
    df_copy.reset_index(inplace = True)
    df_copy[col_name + '_encoder'] = df_copy.index
    train_df = pd.merge(train_df, df_copy[[col_name, col_name + '_encoder']], on=col_name, how='left')
    test_df = pd.merge(test_df, df_copy[[col_name, col_name + '_encoder']], on=col_name, how='left')
    return train_df, test_df

# 导出预测结果
def exportDf(df, fileName):
    df.to_csv('../temp/%s.csv' % fileName, header=True, index=False)

def main():

    # 数据处理
    print('~~~~~~~~~~~~~~开始特征工程~~~~~~~~~~~~~~~~~~~')
    chusai_QY_xinxi_train_df = pd.read_csv('../data/chusai/train/企业基本信息&高管信息&投资信息.csv')
    chusai_QY_xinxi_train_df['is_chusai'] = 1
    fusai_QY_xinxi_train_df = pd.read_csv('../data/fusai/train/企业基本信息&高管信息&投资信息.csv')
    fusai_QY_xinxi_train_df['is_chusai'] = 0
    QY_xinxi_train_df = pd.concat([chusai_QY_xinxi_train_df, fusai_QY_xinxi_train_df])
    QY_xinxi_train_df.rename(columns={'企业名称':'qymc', '注册号':'zch', '统一社会信用代码':'tyshxydm', '注册资金':'zczj', '注册资本(金)币种名称':'zczbbzmc',
                                      '企业(机构)类型名称':'qyjglxmc', '行业门类代码':'hymldm', '成立日期':'clrq', '核准日期':'hzrq', '住所所在地省份':'zsszdsf',
                                      '姓名':'xm', '法定代表人标志':'fddbrbz', '首席代表标志':'sxdbbz', '职务':'zw', '投资人':'tzr', '出资比例':'czbl'}, inplace=True)
    train_df = QY_xinxi_train_df[['qymc', 'zch', 'tyshxydm', 'zczj', 'zczbbzmc', 'qyjglxmc', 'hymldm', 'clrq', 'hzrq', 'zsszdsf', 'is_chusai']]
    train_df.drop_duplicates(['qymc'], keep='first', inplace=True)

    QY_xinxi_test_df = pd.read_csv('../data/fusai/test/企业基本信息&高管信息&投资信息.csv')
    QY_xinxi_test_df.rename(columns={'企业名称':'qymc', '注册号':'zch', '统一社会信用代码':'tyshxydm', '注册资金':'zczj', '注册资本(金)币种名称':'zczbbzmc',
                                      '企业(机构)类型名称':'qyjglxmc', '行业门类代码':'hymldm', '成立日期':'clrq', '核准日期':'hzrq', '住所所在地省份':'zsszdsf',
                                      '姓名':'xm', '法定代表人标志':'fddbrbz', '首席代表标志':'sxdbbz', '职务':'zw', '投资人':'tzr', '出资比例':'czbl'}, inplace=True)
    test_df = QY_xinxi_test_df[['qymc', 'zch', 'tyshxydm', 'zczj', 'zczbbzmc', 'qyjglxmc', 'hymldm', 'clrq', 'hzrq', 'zsszdsf']]
    test_df.drop_duplicates(['qymc'], keep='first', inplace=True)

    #开始给训练集打标签
    #是否失信
    chusai_SXQY_train_df = pd.read_csv('../data/chusai/train/失信被执行人名单.csv')
    fusai_SXQY_train_df = pd.read_csv('../data/fusai/train/失信被执行人名单.csv')
    SXQY_train_df = pd.concat([chusai_SXQY_train_df, fusai_SXQY_train_df])
    SXQY_train_df.rename(columns={'企业名称':'qymc'}, inplace=True)
    SXQY_set = set(SXQY_train_df['qymc'])
    train_df['is_shixin'] = train_df['qymc'].map(lambda x : 1 if x in SXQY_set else 0)

    #是否处罚
    chusai_SGS_FRCF_train_df = pd.read_csv('../data/chusai/train/双公示-法人行政处罚信息.csv')
    fusai_SGS_FRCF_train_df = pd.read_csv('../data/fusai/train/双公示-法人行政处罚信息.csv')
    SGS_FRCF_train_df = pd.concat([chusai_SGS_FRCF_train_df, fusai_SGS_FRCF_train_df])
    SGS_FRCF_train_df.rename(columns={'企业名称':'qymc'}, inplace=True)
    SGS_FRCF_set = set(SGS_FRCF_train_df['qymc'])
    train_df['is_chufa'] = train_df['qymc'].map(lambda x : 1 if x in SGS_FRCF_set else 0)

    train_df = get_QY_czbl_statistic_fea(train_df, QY_xinxi_train_df)
    test_df = get_QY_czbl_statistic_fea(test_df, QY_xinxi_test_df)

    train_df = get_QY_clhzrq_feature(train_df)
    test_df = get_QY_clhzrq_feature(test_df)

    train_df = get_QY_tzrgs_feature(train_df, QY_xinxi_train_df)
    test_df = get_QY_tzrgs_feature(test_df, QY_xinxi_test_df)

    train_df = get_QY_zwzls_feature(train_df, QY_xinxi_train_df)
    test_df = get_QY_zwzls_feature(test_df, QY_xinxi_test_df)

    train_df = get_QY_dbgs_feature(train_df, QY_xinxi_train_df)
    test_df = get_QY_dbgs_feature(test_df, QY_xinxi_test_df)

    train_df = get_QYJL_number_feature(train_df, QY_xinxi_train_df)
    test_df = get_QYJL_number_feature(test_df, QY_xinxi_test_df)

    #给训练集和测试集处理分支机构信息
    chusai_FZJGXX_train_df = pd.read_csv('../data/chusai/train/分支机构信息.csv')
    fusai_FZJGXX_train_df = pd.read_csv('../data/fusai/train/分支机构信息.csv')
    FZJGXX_train_df = pd.concat([chusai_FZJGXX_train_df, fusai_FZJGXX_train_df])
    FZJGXX_train_df.fillna(-1, inplace=True)
    FZJGXX_train_df.rename(columns={'企业名称':'qymc', '分支机构省份':'fzjgsf', '分支企业名称':'fzqymc', '分支机构状态':'fzjgzt',
                                   '分支成立时间':'fzclsj', '分支死亡时间':'fzswsj', '分支行业门类':'fzhyml', '分支行业代码':'fzhydm',
                                   '分支机构区县':'fzjgqx', '分支机构类型':'fzjglx'}, inplace=True)
    FZJGXX_train_df.drop_duplicates(['qymc'], keep='first', inplace=True)
    train_df = pd.merge(train_df, FZJGXX_train_df, on='qymc', how='left')

    chusai_FZJGXX_test_df = pd.read_csv('../data/chusai/test/分支机构信息.csv')
    fusai_FZJGXX_test_df = pd.read_csv('../data/fusai/test/分支机构信息.csv')
    FZJGXX_test_df = pd.concat([chusai_FZJGXX_test_df, fusai_FZJGXX_test_df])
    FZJGXX_test_df.fillna(-1, inplace=True)
    FZJGXX_test_df.rename(columns={'企业名称':'qymc', '分支机构省份':'fzjgsf', '分支企业名称':'fzqymc', '分支机构状态':'fzjgzt',
                                   '分支成立时间':'fzclsj', '分支死亡时间':'fzswsj', '分支行业门类':'fzhyml', '分支行业代码':'fzhydm',
                                   '分支机构区县':'fzjgqx', '分支机构类型':'fzjglx'}, inplace=True)
    FZJGXX_test_df.drop_duplicates(['qymc'], keep='first', inplace=True)
    test_df = pd.merge(test_df, FZJGXX_test_df, on='qymc', how='left')

    train_df = get_FZ_clsw_feature(train_df)
    test_df = get_FZ_clsw_feature(test_df)

    #处理跟机构设立登记信息相关
    chusai_JGSLDJXX_train_df = pd.read_csv('../data/chusai/train/机构设立（变更）登记信息.csv')
    fusai_JGSLDJXX_train_df = pd.read_csv('../data/fusai/train/机构设立（变更）登记信息.csv')
    JGSLDJXX_train_df = pd.concat([chusai_JGSLDJXX_train_df, fusai_JGSLDJXX_train_df])
    JGSLDJXX_drop_list = ['机构全称英文', '备注', '设立登记类型', '法定代表人证件名称', '经济类型', '联系电话', '监管单位', '股东(发起人)',
                '资金币种', '交换单位全称', '统一社会信用代码']
    JGSLDJXX_train_df.drop(JGSLDJXX_drop_list, axis=1, inplace=True)
    JGSLDJXX_train_df.fillna(-1, inplace=True)
    JGSLDJXX_train_df.rename(columns={'企业类型代码':'qylxdm', '企业类型名称':'qylxmc', '企业名称':'qymc', '数据状态':'sjzt', '数据来源':'sjly', '创建时间':'cjsj', '创建人ID':'cjrID',
                                      '信息提供部门编码':'xxtgbmbm', '信息提供部门名称':'xxtgbmmc', '提供日期':'tgrq', '任务编号':'rwbh',
                                     '组织机构代码':'zzjgdm', '工商注册号':'gszch', '种类':'zl', '法定代表人姓名':'fddbrxm',
                                      '法定代表人证件号码':'fddbrzjhm', '注册（开办）资金':'zckbzj', '实收资金':'sszj', '经营范围':'jyfw',
                                      '所属行业名称':'sshymc', '所属行业代码':'sshydm', '机构地址（住所）':'jgdzzs', '发证机关名称':'fzjgmc',
                                     '发证日期':'fzrq', '（变更）核准日期':'bghzrq', '行政区划':'xzgh', '企业经度':'qyjd', '企业纬度':'qywd',
                                     '是否有经纬度':'sfyjwd', '企业地址是否有变化':'qydzsfybh'}, inplace=True)
    JGSLDJXX_train_df.drop_duplicates(['qymc'], keep='first', inplace=True)
    train_df = pd.merge(train_df, JGSLDJXX_train_df, on='qymc', how='left')

    JGSLDJXX_test_df = pd.read_csv('../data/fusai/test/机构设立（变更）登记信息.csv')
    JGSLDJXX_test_df.drop(JGSLDJXX_drop_list, axis=1, inplace=True)
    JGSLDJXX_test_df.fillna(-1, inplace=True)
    JGSLDJXX_test_df.rename(columns={'企业类型代码':'qylxdm', '企业类型名称':'qylxmc', '企业名称':'qymc', '数据状态':'sjzt', '数据来源':'sjly', '创建时间':'cjsj', '创建人ID':'cjrID',
                                      '信息提供部门编码':'xxtgbmbm', '信息提供部门名称':'xxtgbmmc', '提供日期':'tgrq', '任务编号':'rwbh',
                                     '组织机构代码':'zzjgdm', '工商注册号':'gszch', '种类':'zl', '法定代表人姓名':'fddbrxm',
                                      '法定代表人证件号码':'fddbrzjhm', '注册（开办）资金':'zckbzj', '实收资金':'sszj', '经营范围':'jyfw',
                                      '所属行业名称':'sshymc', '所属行业代码':'sshydm', '机构地址（住所）':'jgdzzs', '发证机关名称':'fzjgmc',
                                     '发证日期':'fzrq', '（变更）核准日期':'bghzrq', '行政区划':'xzgh', '企业经度':'qyjd', '企业纬度':'qywd',
                                     '是否有经纬度':'sfyjwd', '企业地址是否有变化':'qydzsfybh'}, inplace=True)
    JGSLDJXX_test_df.drop_duplicates(['qymc'], keep='first', inplace=True)
    test_df = pd.merge(test_df, JGSLDJXX_test_df, on='qymc', how='left')

    my_model = Word2Vec.load('../temp/jyfw_w2v_online.model')
    word_wv = my_model.wv

    train_df['jyfw_jieba'] = train_df['jyfw'].map(lambda x : np.nan if x is np.nan else jieba_sentences(str(x)))
    test_df['jyfw_jieba'] = test_df['jyfw'].map(lambda x : np.nan if x is np.nan else jieba_sentences(str(x)))

    dir_num_features = 10
    train_df['jyfw_array'] = train_df['jyfw_jieba'].map(lambda x : np.zeros(dir_num_features) if x is np.nan else get_dir_w2v_array(x, word_wv, dir_num_features))
    test_df['jyfw_array'] = test_df['jyfw_jieba'].map(lambda x : np.zeros(dir_num_features) if x is np.nan else get_dir_w2v_array(x, word_wv, dir_num_features))

    train_df = get_jyfw_w2v_feature(train_df, dir_num_features)
    test_df = get_jyfw_w2v_feature(test_df, dir_num_features)

    chusai_FRXZXKZX_train_df = pd.read_csv('../data/chusai/train/法人行政许可注（撤、吊）销信息.csv')
    fusai_FRXZXKZX_train_df = pd.read_csv('../data/fusai/train/法人行政许可注（撤、吊）销信息.csv')
    FRXZXKZX_train_df = pd.concat([chusai_FRXZXKZX_train_df, fusai_FRXZXKZX_train_df])
    FRXZXKZX_test_df = pd.read_csv('../data/fusai/test/法人行政许可注（撤、吊）销信息.csv')

    train_df = get_FRXZXKZX_number_feature(train_df, FRXZXKZX_train_df)
    test_df = get_FRXZXKZX_number_feature(test_df, FRXZXKZX_test_df)

    chusai_QYBZRYXX_train_df = pd.read_csv('../data/chusai/train/企业表彰荣誉信息.csv')
    fusai_QYBZRYXX_train_df = pd.read_csv('../data/fusai/train/企业表彰荣誉信息.csv')
    QYBZRYXX_train_df = pd.concat([chusai_QYBZRYXX_train_df, fusai_QYBZRYXX_train_df])
    QYBZRYXX_test_df = pd.read_csv('../data/fusai/test/企业表彰荣誉信息.csv')

    train_df = get_QYBZRYXX_number_feature(train_df, QYBZRYXX_train_df)
    test_df = get_QYBZRYXX_number_feature(test_df, QYBZRYXX_test_df)

    chusai_QYFZCHRD_train_df = pd.read_csv('../data/chusai/train/企业非正常户认定.csv')
    fusai_QYFZCHRD_train_df = pd.read_csv('../data/fusai/train/企业非正常户认定.csv')
    QYFZCHRD_train_df = pd.concat([chusai_QYFZCHRD_train_df, fusai_QYFZCHRD_train_df])
    QYFZCHRD_test_df = pd.read_csv('../data/fusai/test/企业非正常户认定.csv')
    QYFZCHRD_drop_list = ['创建人ID', '机构全称英文', '机构全称中文', '工商注册号', '统一社会信用代码', '备注', '数据状态']
    QYFZCHRD_train_df.drop(QYFZCHRD_drop_list, axis=1, inplace=True)
    QYFZCHRD_test_df.drop(QYFZCHRD_drop_list, axis=1, inplace=True)
    QYFZCHRD_train_df.rename(columns={'企业名称':'qymc', '关联机构设立登记表主键ID':'gljgsldjbzjID', '数据来源':'sjly',
                                     '创建时间':'cjsj', '信息提供部门编码':'xxtgbmbm', '信息提供部门名称':'xxtgbmmc',
                                     '提供日期':'tgrq', '任务编号':'rwbh', '组织机构代码':'zzjgdm', '税务管理码':'swglm',
                                     '纳税人识别号':'nsrsbh', '纳税人状态':'nsrzt', '法定代表人姓名':'fddbrxm',
                                     '登记注册类型':'djzclx', '注册地址':'zcdz', '认定日期':'rdrq', '应用年限':'yynx',
                                     '管理机构':'gljg'}, inplace=True)
    QYFZCHRD_test_df.rename(columns={'企业名称':'qymc', '关联机构设立登记表主键ID':'gljgsldjbzjID', '数据来源':'sjly',
                                     '创建时间':'cjsj', '信息提供部门编码':'xxtgbmbm', '信息提供部门名称':'xxtgbmmc',
                                     '提供日期':'tgrq', '任务编号':'rwbh', '组织机构代码':'zzjgdm', '税务管理码':'swglm',
                                     '纳税人识别号':'nsrsbh', '纳税人状态':'nsrzt', '法定代表人姓名':'fddbrxm',
                                     '登记注册类型':'djzclx', '注册地址':'zcdz', '认定日期':'rdrq', '应用年限':'yynx',
                                     '管理机构':'gljg'}, inplace=True)

    train_df = get_QYFZCHRD_number_feature(train_df, QYFZCHRD_train_df)
    test_df = get_QYFZCHRD_number_feature(test_df, QYFZCHRD_test_df)

    train_df = get_QYFZCHRD_basic_feature(train_df, QYFZCHRD_train_df)
    test_df = get_QYFZCHRD_basic_feature(test_df, QYFZCHRD_test_df)

    chusai_QYSWDJXX_train_df = pd.read_csv('../data/chusai/train/企业税务登记信息.csv')
    fusai_QYSWDJXX_train_df = pd.read_csv('../data/fusai/train/企业税务登记信息.csv')
    QYSWDJXX_train_df = pd.concat([chusai_QYSWDJXX_train_df, fusai_QYSWDJXX_train_df])
    QYSWDJXX_test_df = pd.read_csv('../data/fusai/test/企业税务登记信息.csv')
    QYSWDJXX_drop_list = ['任务编号', '变更日期', '数据状态', '创建人ID', '机构全称英文', '备注', '统一社会信用代码', '信息提供部门编码', '信息提供部门名称', '工商注册号']
    QYSWDJXX_train_df.drop(QYSWDJXX_drop_list, axis=1, inplace=True)
    QYSWDJXX_test_df.drop(QYSWDJXX_drop_list, axis=1, inplace=True)
    QYSWDJXX_train_df.rename(columns={'企业名称':'qymc', '关联机构设立登记表主键ID':'gljgsldjbzjID', '数据来源':'sjly', '创建时间':'cjsj',
                                     '提供日期':'tgrq', '组织机构代码':'zzjgdm', '税务管理码':'swglm', '纳税人识别号':'nsrsbh', '法定代表人姓名':'fddbrxm',
                                     '法定代表人证件名称':'fddbrzjmc', '登记注册类型':'djzclx', '审核结果':'shjg', '审核时间':'shsj',
                                     '审核单位':'shdw', '区域':'qy'}, inplace=True)
    QYSWDJXX_test_df.rename(columns={'企业名称':'qymc', '关联机构设立登记表主键ID':'gljgsldjbzjID', '数据来源':'sjly', '创建时间':'cjsj',
                                     '提供日期':'tgrq', '组织机构代码':'zzjgdm', '税务管理码':'swglm', '纳税人识别号':'nsrsbh', '法定代表人姓名':'fddbrxm',
                                     '法定代表人证件名称':'fddbrzjmc', '登记注册类型':'djzclx', '审核结果':'shjg', '审核时间':'shsj',
                                     '审核单位':'shdw', '区域':'qy'}, inplace=True)

    train_df = get_QYSWDJXX_number_feature(train_df, QYSWDJXX_train_df)
    test_df = get_QYSWDJXX_number_feature(test_df, QYSWDJXX_test_df)

    chusai_SDBDJQQJMCF_train_df = pd.read_csv('../data/chusai/train/双打办打击侵权假冒处罚案件信息.csv')
    fusai_SDBDJQQJMCF_train_df = pd.read_csv('../data/fusai/train/双打办打击侵权假冒处罚案件信息.csv')
    SDBDJQQJMCF_train_df = pd.concat([chusai_SDBDJQQJMCF_train_df, fusai_SDBDJQQJMCF_train_df])
    SDBDJQQJMCF_test_df = pd.read_csv('../data/fusai/test/双打办打击侵权假冒处罚案件信息.csv')
    SDBDJQQJMCF_drop_list = ['数据状态', '数据来源', '创建人ID', '信息提供部门编码', '信息提供部门名称', '任务编号', '关联机构设立登记表主键ID', '被处罚的自然人姓名', '被处罚的自然人身份证号']
    SDBDJQQJMCF_train_df.drop(SDBDJQQJMCF_drop_list, axis=1, inplace=True)
    SDBDJQQJMCF_test_df.drop(SDBDJQQJMCF_drop_list, axis=1, inplace=True)
    SDBDJQQJMCF_train_df.rename(columns={'企业名称':'qymc', '创建时间':'cjsj', '提供日期':'tgrq', '行政处罚决定书文号':'xzcfjdswh',
                                        '被处罚企业统一社会信用编码':'bcfqytyshxybm', '被处罚企业工商注册号':'bcfqygszch', '被处罚的企业法定代表人姓名':'bcfdqyfddbrxm',
                                        '被处罚的企业法定代表人身份证号':'bcfdqyfddbrsfzh', '违反法律、法规或规章的主要事实':'wfflfghgzdzyss',
                                        '行政处罚的种类和依据':'xzcfdzlhyj', '行政处罚的履行方式和期限':'xzcfdlxfshqx', '作出处罚决定的行政执法机关名称':'zccfjddxzzfjgmc',
                                        '作出处罚决定的日期':'zccfjddrq', '公布方式及网址':'gbfsjwz'}, inplace=True)
    SDBDJQQJMCF_test_df.rename(columns={'企业名称':'qymc', '创建时间':'cjsj', '提供日期':'tgrq', '行政处罚决定书文号':'xzcfjdswh',
                                        '被处罚企业统一社会信用编码':'bcfqytyshxybm', '被处罚企业工商注册号':'bcfqygszch', '被处罚的企业法定代表人姓名':'bcfdqyfddbrxm',
                                        '被处罚的企业法定代表人身份证号':'bcfdqyfddbrsfzh', '违反法律、法规或规章的主要事实':'wfflfghgzdzyss',
                                        '行政处罚的种类和依据':'xzcfdzlhyj', '行政处罚的履行方式和期限':'xzcfdlxfshqx', '作出处罚决定的行政执法机关名称':'zccfjddxzzfjgmc',
                                        '作出处罚决定的日期':'zccfjddrq', '公布方式及网址':'gbfsjwz'}, inplace=True)

    train_df = get_SDBDJQQJMCF_number_feature(train_df, SDBDJQQJMCF_train_df)
    test_df = get_SDBDJQQJMCF_number_feature(test_df, SDBDJQQJMCF_test_df)

    chusai_SGSFRXZXKXX_train_df = pd.read_csv('../data/chusai/train/双公示-法人行政许可信息.csv')
    fusai_SGSFRXZXKXX_train_df = pd.read_csv('../data/fusai/train/双公示-法人行政许可信息.csv')
    SGSFRXZXKXX_train_df = pd.concat([chusai_SGSFRXZXKXX_train_df, fusai_SGSFRXZXKXX_train_df])
    SGSFRXZXKXX_test_df = pd.read_csv('../data/fusai/test/双公示-法人行政许可信息.csv')
    SGSFRXZXKXX_drop_list = ['自动上报信用中国状态', '自动上报时间', '错误原因', '接口返回信用中国的ID', '任务编号', '机构全称英文', '备注', '行政相对人税务登记号']
    SGSFRXZXKXX_train_df.drop(SGSFRXZXKXX_drop_list, axis=1, inplace=True)
    SGSFRXZXKXX_test_df.drop(SGSFRXZXKXX_drop_list, axis=1, inplace=True)
    # print(SGSFRXZXKXX_train_df.nunique())
    SGSFRXZXKXX_train_df.rename(columns={'企业名称':'qymc', '关联机构设立登记表主键ID':'gljgsldjbzjID', '进入业务库的时间':'jrywkdsj',
                                        '数据状态_1':'sjzt1', '数据来源':'sjly', '创建时间':'cjsj', '创建人ID':'cjrID', '信息提供部门编码':'xxtgbmbm',
                                        '信息提供部门名称':'xxtgbmmc', '提供日期':'tgrq', '行政相对人代码_2':'xzxdrdm2', '行政相对人代码＿3':'xzxdrdm3',
                                        '行政相对人代码_1':'xzxdrdm1', '许可决定书文号':'xkjdswh', '项目名称':'xmmc', '行政许可编码':'xzxkbm', '审批类别':'splb',
                                         '许可内容':'xknr', '行政相对人名称':'xzxdrmc', '行政相对人代码＿4':'xzxdrdm4', '法定代表人名称':'fddbrmc',
                                         '行政相对人代码＿5':'xzxdrdm5', '许可决定日期':'xkjdrq', '许可截止期':'xkjzq', '许可机关':'xkjg', '数据状态_2':'sjzt2',
                                        '地方编码':'dfbm', '数据更新时间戳':'sjgxsjc', '信息使用范围':'xysyfw', '公示截止期':'gsjzq'}, inplace=True)
    SGSFRXZXKXX_test_df.rename(columns={'企业名称':'qymc', '关联机构设立登记表主键ID':'gljgsldjbzjID', '进入业务库的时间':'jrywkdsj',
                                        '数据状态_1':'sjzt1', '数据来源':'sjly', '创建时间':'cjsj', '创建人ID':'cjrID', '信息提供部门编码':'xxtgbmbm',
                                        '信息提供部门名称':'xxtgbmmc', '提供日期':'tgrq', '行政相对人代码_2':'xzxdrdm2', '行政相对人代码＿3':'xzxdrdm3',
                                        '行政相对人代码_1':'xzxdrdm1', '许可决定书文号':'xkjdswh', '项目名称':'xmmc', '行政许可编码':'xzxkbm', '审批类别':'splb',
                                         '许可内容':'xknr', '行政相对人名称':'xzxdrmc', '行政相对人代码＿4':'xzxdrdm4', '法定代表人名称':'fddbrmc',
                                         '行政相对人代码＿5':'xzxdrdm5', '许可决定日期':'xkjdrq', '许可截止期':'xkjzq', '许可机关':'xkjg', '数据状态_2':'sjzt2',
                                        '地方编码':'dfbm', '数据更新时间戳':'sjgxsjc', '信息使用范围':'xysyfw', '公示截止期':'gsjzq'}, inplace=True)

    train_df = get_SGSFRXZXKXX_number_feature(train_df, SGSFRXZXKXX_train_df)
    test_df = get_SGSFRXZXKXX_number_feature(test_df, SGSFRXZXKXX_test_df)

    train_df = get_FRXZXK_last_statistic_feature(train_df, SGSFRXZXKXX_train_df)
    test_df = get_FRXZXK_last_statistic_feature(test_df, SGSFRXZXKXX_test_df)

    chusai_XKZZNJXX_train_df = pd.read_csv('../data/chusai/train/许可资质年检信息.csv')
    fusai_XKZZNJXX_train_df = pd.read_csv('../data/fusai/train/许可资质年检信息.csv')
    XKZZNJXX_train_df = pd.concat([chusai_XKZZNJXX_train_df, fusai_XKZZNJXX_train_df])
    XKZZNJXX_test_df = pd.read_csv('../data/fusai/test/许可资质年检信息.csv')
    XKZZNJXX_drop_list = ['数据状态', '数据来源', '创建人ID', '任务编号', '机构全称英文', '备注', '权力名称', '权力编码', '统一社会信用代码']
    XKZZNJXX_train_df.drop(XKZZNJXX_drop_list, axis=1, inplace=True)
    XKZZNJXX_test_df.drop(XKZZNJXX_drop_list, axis=1, inplace=True)
    # print(XKZZNJXX_train_df.nunique())
    XKZZNJXX_train_df.rename(columns={'企业名称':'qymc', '关联机构设立登记表主键ID':'gljgsldjbzjID', '创建时间':'cjsj', '信息提供部门编码':'xxtgbmbm',
                                     '信息提供部门名称':'xxtgbmmc', '提供日期':'tgrq', '组织机构代码':'zzjgdm', '工商注册号':'gszch', '证书编号':'zsbh',
                                     '年检年度':'njnd', '年检结果':'njjg', '年检机关全称':'njjgmc', '年检事项名称':'njsxmc', '年检日期':'njrq', '交换单位全称':'jhdwqc'}, inplace=True)
    XKZZNJXX_test_df.rename(columns={'企业名称':'qymc', '关联机构设立登记表主键ID':'gljgsldjbzjID', '创建时间':'cjsj', '信息提供部门编码':'xxtgbmbm',
                                     '信息提供部门名称':'xxtgbmmc', '提供日期':'tgrq', '组织机构代码':'zzjgdm', '工商注册号':'gszch', '证书编号':'zsbh',
                                     '年检年度':'njnd', '年检结果':'njjg', '年检机关全称':'njjgmc', '年检事项名称':'njsxmc', '年检日期':'njrq', '交换单位全称':'jhdwqc'}, inplace=True)

    train_df = get_XKZZNJXX_number_feature(train_df, XKZZNJXX_train_df)
    test_df = get_XKZZNJXX_number_feature(test_df, XKZZNJXX_test_df)

    train_df = get_XKZZNJXX_last_statistic_feature(train_df, XKZZNJXX_train_df)
    test_df = get_XKZZNJXX_last_statistic_feature(test_df, XKZZNJXX_test_df)

    chusai_ZPSJ_train_df = pd.read_csv('../data/chusai/train/招聘数据.csv')
    fusai_ZPSJ_train_df = pd.read_csv('../data/fusai/train/招聘数据.csv')
    ZPSJ_train_df = pd.concat([chusai_ZPSJ_train_df, fusai_ZPSJ_train_df])
    ZPSJ_test_df = pd.read_csv('../data/fusai/test/招聘数据.csv')
    ZPSJ_train_df.rename(columns={'企业名称':'qymc', '网站名称':'wzmc', '工作经验':'gzjy', '工作地点':'gzdd', '职位类别':'zylb',
                                 '招聘人数':'zprs', '职位月薪':'zwyx', '最低学历':'zdxl', '业务主键':'ywzj', '招聘日期':'zprq'}, inplace=True)
    ZPSJ_test_df.rename(columns={'企业名称':'qymc', '网站名称':'wzmc', '工作经验':'gzjy', '工作地点':'gzdd', '职位类别':'zylb',
                                 '招聘人数':'zprs', '职位月薪':'zwyx', '最低学历':'zdxl', '业务主键':'ywzj', '招聘日期':'zprq'}, inplace=True)
    train_df = get_ZPSJ_number_feature(train_df, ZPSJ_train_df)
    test_df = get_ZPSJ_number_feature(test_df, ZPSJ_test_df)

    train_df = get_ZPSJ_kind_feature(train_df, ZPSJ_train_df)
    test_df = get_ZPSJ_kind_feature(test_df, ZPSJ_test_df)

    train_df = get_ZPSJ_zprs_statistic_feature(train_df, ZPSJ_train_df)
    test_df = get_ZPSJ_zprs_statistic_feature(test_df, ZPSJ_test_df)

    chusai_ZZDJBGXX_train_df = pd.read_csv('../data/chusai/train/资质登记（变更）信息.csv')
    fusai_ZZDJBGXX_train_df = pd.read_csv('../data/fusai/train/资质登记（变更）信息.csv')
    ZZDJBGXX_train_df = pd.concat([chusai_ZZDJBGXX_train_df, fusai_ZZDJBGXX_train_df])
    ZZDJBGXX_test_df = pd.read_csv('../data/fusai/test/资质登记（变更）信息.csv')
    ZZDJBGXX_drop_list = ['数据状态', '创建人ID', '数据来源', '创建人ID', '任务编号', '机构全称英文', '统一社会信用代码',
                          '备注', '资质等级', '交换单位全称', '权力名称', '权力编码', '变更核准日期', '种类']
    ZZDJBGXX_train_df.drop(ZZDJBGXX_drop_list, axis=1, inplace=True)
    ZZDJBGXX_test_df.drop(ZZDJBGXX_drop_list, axis=1, inplace=True)
    # print(ZZDJBGXX_train_df.nunique())
    ZZDJBGXX_train_df.rename(columns={'企业名称':'qymc', '关联机构设立登记表主键ID':'gljgsldjbzjID', '创建时间':'cjsj', '信息提供部门编码':'xxtgbmbm',
                                     '信息提供部门名称':'xxtgbmmc', '提供日期':'tgrq', '组织机构代码':'zzjgdm', '工商注册号':'gszch', '资质证书编号':'zzzsbh',
                                     '资质名称':'zzmc', '执业范围':'zyfw', '资质生效期':'zzsxq', '资质截止期':'zzjzq', '认定机关全称':'rdjgqc',
                                     '认定日期':'rdrq'}, inplace=True)
    ZZDJBGXX_test_df.rename(columns={'企业名称':'qymc', '关联机构设立登记表主键ID':'gljgsldjbzjID', '创建时间':'cjsj', '信息提供部门编码':'xxtgbmbm',
                                     '信息提供部门名称':'xxtgbmmc', '提供日期':'tgrq', '组织机构代码':'zzjgdm', '工商注册号':'gszch', '资质证书编号':'zzzsbh',
                                     '资质名称':'zzmc', '执业范围':'zyfw', '资质生效期':'zzsxq', '资质截止期':'zzjzq', '认定机关全称':'rdjgqc',
                                     '认定日期':'rdrq'}, inplace=True)
    train_df = get_ZZDJBGXX_number_feature(train_df, ZZDJBGXX_train_df)
    test_df = get_ZZDJBGXX_number_feature(test_df, ZZDJBGXX_test_df)

    train_df = get_ZZDJBGXX_last_statistic_feature(train_df, ZZDJBGXX_train_df)
    test_df = get_ZZDJBGXX_last_statistic_feature(test_df, ZZDJBGXX_test_df)

    train_df['jgdzzs_top3'] = train_df['jgdzzs'].map(lambda x : np.nan if x is np.nan else x[:3] if len(x) >=3 else x)
    train_df['jgdzzs_top6'] = train_df['jgdzzs'].map(lambda x : np.nan if x is np.nan else x[:6] if len(x) >=6 else x)
    test_df['jgdzzs_top3'] = test_df['jgdzzs'].map(lambda x : np.nan if x is np.nan else x[:3] if len(x) >=3 else x)
    test_df['jgdzzs_top6'] = test_df['jgdzzs'].map(lambda x : np.nan if x is np.nan else x[:6] if len(x) >=6 else x)

    temp_df = pd.concat([train_df, test_df])
    # labelencoder 转化
    labelEncoder_columns_list = ['sshydm', 'tyshxydm', 'zczbbzmc', 'hymldm', 'zsszdsf', 'fzhyml', 'xxtgbmmc', 'zl', 'sszj', 'fzjgmc', 'qyjglxmc', 'qylxmc',
                                'QYFZCHRD_swglm', 'QYFZCHRD_gljg', 'QYFZCHRD_djzclx',
                                'SGSFRXZXKXX_xknr', 'SGSFRXZXKXX_xmmc', 'SGSFRXZXKXX_splb', 'SGSFRXZXKXX_xkjg', 'SGSFRXZXKXX_xysyfw', 'SGSFRXZXKXX_sjzt2',
                                'XKZZNJXX_njnd', 'XKZZNJXX_njjgmc', 'XKZZNJXX_njsxmc',
                                'ZZDJBGXX_zzmc', 'ZZDJBGXX_zyfw', 'ZZDJBGXX_rdjgqc', 'ZZDJBGXX_zzzsbh',
                                'jgdzzs', 'jyfw', 'sshymc', 'fzrq', 'bghzrq', 'gszch', 'fddbrzjhm', 'zzjgdm',
                                'SGSFRXZXKXX_xzxkbm', 'SGSFRXZXKXX_xkjdswh', 'SGSFRXZXKXX_xxtgbmmc', 'XKZZNJXX_xxtgbmmc',
                                'QYFZCHRD_rwbh', 'QYFZCHRD_zcdz', 'QYFZCHRD_xxtgbmmc',
                                 'jgdzzs_top3', 'jgdzzs_top6',
                                'QYSWDJXX_fddbrzjmc', 'QYSWDJXX_shdw', 'QYSWDJXX_qy', 'QYSWDJXX_shjg', 'QYSWDJXX_djzclx', ]
    for fea in labelEncoder_columns_list:
        train_df, test_df = get_type_future_encoder(train_df, test_df, fea)
        print(fea + " finish!!!")

    exportDf(train_df, 'fusai_train_df_all')
    exportDf(test_df, 'fusai_test_df_all')

    print('~~~~~~~~~~~~~~完毕~~~~~~~~~~~~~~~~~~~')


if __name__ == '__main__':
    main()
