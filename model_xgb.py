#!/usr/bin/env python
# -*-coding:utf-8-*-

import pandas as pd
import numpy as np
from datetime import *
from sklearn.preprocessing import *
import xgboost as xgb 
from sklearn.model_selection import train_test_split, KFold, GridSearchCV, StratifiedKFold
import warnings

warnings.filterwarnings("ignore")

# xgboost模型
class XgbModel:
    def __init__(self, feaNames=None, params={}):
        self.feaNames = feaNames
        self.params = {
            'objective': 'binary:logistic',
            'eval_metric':'auc',
            'silent': True,
            'eta': 0.05,
            'max_depth': 6,
            'gamma': 1,
            'subsample': 0.9,
            'colsample_bytree': 0.95,
            'min_child_weight': 3,
            'max_delta_step': 1,
            'lambda': 30,
        }
        for k,v in params.items():
            self.params[k] = v
        self.clf = None

    def train(self, X, y, train_size=1, test_size=0.1, verbose=True, num_boost_round=1000, early_stopping_rounds=3):
        X = X.astype(float)
        if train_size==1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
            X_train, y_train = X, y
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size)
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feaNames)
        dval = xgb.DMatrix(X_test, label=y_test, feature_names=self.feaNames)
        watchlist = [(dtrain,'train'),(dval,'val')]
        clf = xgb.train(
            self.params, dtrain, 
            num_boost_round = num_boost_round, 
            evals = watchlist, 
            early_stopping_rounds = early_stopping_rounds,
            verbose_eval=verbose
        )
        self.clf = clf

    def trainCV(self, X, y, nFold=10, verbose=True, num_boost_round=8000, early_stopping_rounds=100):
        X = X.astype(float)
        dtrain = xgb.DMatrix(X, label=y, feature_names=self.feaNames)
        cvResult = xgb.cv(
            self.params, dtrain, 
            num_boost_round = num_boost_round, 
            nfold = nFold,
            early_stopping_rounds = early_stopping_rounds,
            verbose_eval=verbose
        )
        clf = xgb.train(
            self.params, dtrain, 
            num_boost_round = cvResult.shape[0], 
        )
        self.clf = clf

    def predict(self, X):
        X = X.astype(float)
        return self.clf.predict(xgb.DMatrix(X, feature_names=self.feaNames))

    def getFeaScore(self, show=False):
        fscore = self.clf.get_score()
        feaNames = fscore.keys()
        scoreDf = pd.DataFrame(index=feaNames, columns=['importance'])
        for k,v in fscore.items():
            scoreDf.loc[k, 'importance'] = v
        if show:
            print(scoreDf.sort_index(by=['importance'], ascending=False))
        return scoreDf

# 导出预测结果
def exportResult(df, fileName):
    df.to_csv('../temp/%s.csv' % fileName, header=True, index=False)

if __name__ == "__main__":
    # 导入特征文件
    print('导入数据！')
    train_df = pd.read_csv('../temp/fusai_train_df_all.csv')
    test_df = pd.read_csv('../temp/fusai_test_df_all.csv')
    
    #################################################################################
    chusai_QY_xinxi_train_df = pd.read_csv('../data/chusai/train/企业基本信息&高管信息&投资信息.csv')
    fusai_QY_xinxi_train_df = pd.read_csv('../data/fusai/train/企业基本信息&高管信息&投资信息.csv')
    QY_xinxi_train_df = pd.concat([chusai_QY_xinxi_train_df, fusai_QY_xinxi_train_df])
    QY_xinxi_train_df.rename(columns={'企业名称':'qymc', '注册号':'zch', '统一社会信用代码':'tyshxydm', '注册资金':'zczj', '注册资本(金)币种名称':'zczbbzmc', 
                    '企业(机构)类型名称':'qyjglxmc', '行业门类代码':'hymldm', '成立日期':'clrq', '核准日期':'hzrq', '住所所在地省份':'zsszdsf', 
                    '姓名':'xm', '法定代表人标志':'fddbrbz', '首席代表标志':'sxdbbz', '职务':'zw', '投资人':'tzr', '出资比例':'czbl'}, inplace=True)
    after_train_df = QY_xinxi_train_df[['qymc', 'zch', 'tyshxydm', 'zczj', 'zczbbzmc', 'qyjglxmc', 'hymldm', 'clrq', 'hzrq', 'zsszdsf']]
    after_train_df.drop_duplicates(['qymc'], keep='first', inplace=True)
    train_suzhou_set = set(after_train_df[after_train_df.zsszdsf == '江苏省']['qymc'])
    train_df['is_suzhou'] = train_df['qymc'].map(lambda x : 1 if x in train_suzhou_set else 0)

    QY_xinxi_test_df = pd.read_csv('../data/fusai/test/企业基本信息&高管信息&投资信息.csv')
    QY_xinxi_test_df.rename(columns={'企业名称':'qymc', '注册号':'zch', '统一社会信用代码':'tyshxydm', '注册资金':'zczj', '注册资本(金)币种名称':'zczbbzmc', 
                    '企业(机构)类型名称':'qyjglxmc', '行业门类代码':'hymldm', '成立日期':'clrq', '核准日期':'hzrq', '住所所在地省份':'zsszdsf',
                    '姓名':'xm', '法定代表人标志':'fddbrbz', '首席代表标志':'sxdbbz', '职务':'zw', '投资人':'tzr', '出资比例':'czbl'}, inplace=True)
    after_test_df = QY_xinxi_test_df[['qymc', 'zch', 'tyshxydm', 'zczj', 'zczbbzmc', 'qyjglxmc', 'hymldm', 'clrq', 'hzrq', 'zsszdsf']]
    after_test_df.drop_duplicates(['qymc'], keep='first', inplace=True)
    suzhou_set = set(after_test_df[after_test_df.zsszdsf == '江苏省']['qymc'])
    test_df['is_suzhou'] = test_df['qymc'].map(lambda x : 1 if x in suzhou_set else 0)


    categorical_feature_shixin = ['zsszdsf_encoder', 'qyjglxmc_encoder', 'hymldm_encoder', 
                       'zczbbzmc_encoder', 'fzjglx', 'fzhydm', 'fzhyml_encoder', 'fzjgzt',
                      'fzjgsf', 'qylxmc_encoder', 'xzgh', 'sshydm_encoder', 'zl_encoder', 
                      'SGSFRXZXKXX_xknr_encoder', 'QY_fddbrbz_gs', 'SGSFRXZXKXX_xmmc_encoder']

    categorical_feature_chufa = ['zsszdsf_encoder', 'qyjglxmc_encoder', 'hymldm_encoder', 
                       'zczbbzmc_encoder', 'fzjglx', 'fzhydm', 'fzhyml_encoder', 'fzjgzt',
                      'fzjgsf', 'qylxmc_encoder', 'xzgh', 'sshydm_encoder', 'zl_encoder']
    
    #################################################################################
    FRXZXKZX_test_df = pd.read_csv('../data/fusai/test/法人行政许可注（撤、吊）销信息.csv')
    fr_qymc_set = set(FRXZXKZX_test_df['企业名称'])

    test_df['is_fr'] = test_df['qymc'].map(lambda x : 1 if x in fr_qymc_set else 0)

    QYFZCHRD_test_df = pd.read_csv('../data/fusai/test/企业非正常户认定.csv')
    fzc_qymc_set = set(QYFZCHRD_test_df['企业名称'])

    test_df['is_fzc'] = test_df['qymc'].map(lambda x : 1 if x in fzc_qymc_set else 0)
    
    #################################################################################
    FZJGXX_test_df = pd.read_csv('../data/fusai/test/分支机构信息.csv')
    fz_qymc_set = set(FZJGXX_test_df['企业名称'])

    test_df['is_fz'] = test_df['qymc'].map(lambda x : 1 if x in fz_qymc_set else 0)
    
    #################################################################################
    SDBDJQQJMCF_test_df = pd.read_csv('../data/fusai/test/双打办打击侵权假冒处罚案件信息.csv')
    qq_qymc_set = set(SDBDJQQJMCF_test_df['企业名称'])

    test_df['is_qq'] = test_df['qymc'].map(lambda x : 1 if x in qq_qymc_set else 0)
    


    #################################################################################
    # 失信任务的特征
    shixin_fea = ['zczj', 'fzjgsf', 'fzjgzt', 'fzhydm', 'fzjgqx', 'fzjglx', 'sjly', 'rwbh', 
       'QYFZCHRD_yynx', 'sshydm_encoder', 'xzgh', 'qyjd', 'qywd', 
       'QYBZRYXX_number', 'QYFZCHRD_number', 'QYSWDJXX_number', 'FRXZXKZX_number',
       'SDBDJQQJMCF_number', 'SGSFRXZXKXX_number', 'XKZZNJXX_number',
       'ZPSJ_number', 'ZZDJBGXX_number', 'QYJL_number', 
       'zczbbzmc_encoder', 'hymldm_encoder', 'zsszdsf_encoder', 'fzhyml_encoder',
       'xxtgbmmc_encoder', 'zl_encoder', 'sszj_encoder', 'fzjgmc_encoder',
       'qyjglxmc_encoder', 'qylxmc_encoder', 'QYFZCHRD_swglm_encoder',
       'QYFZCHRD_gljg_encoder', 'QYFZCHRD_djzclx_encoder', 
       'SGSFRXZXKXX_dfbm', 'SGSFRXZXKXX_xknr_encoder', 'SGSFRXZXKXX_xmmc_encoder', 'SGSFRXZXKXX_splb_encoder', 
       'SGSFRXZXKXX_xkjg_encoder', 'SGSFRXZXKXX_xysyfw_encoder', 'SGSFRXZXKXX_sjzt2_encoder', 
       'XKZZNJXX_njnd_encoder', 'XKZZNJXX_njjgmc_encoder', 'XKZZNJXX_njsxmc_encoder', 
       'ZZDJBGXX_zzmc_encoder', 'ZZDJBGXX_zyfw_encoder', 'ZZDJBGXX_rdjgqc_encoder', 'ZZDJBGXX_zzzsbh_encoder', 
       'QY_tzrgs', 'QY_zwzls', 'QY_fddbrbz_gs', 'QY_sxdbbz_gs',  
       'fzsw_deltaDay', 'clhz_deltaDay', 
       'clrq_month', 'clrq_day', 'hzrq_year', 'hzrq_month', 'hzrq_day', 'clrq_year',   
       'ZPSJ_zprs_max', 'ZPSJ_zprs_min', 'ZPSJ_zprs_mean', 'ZPSJ_zprs_std',
       'QYSWDJXX_fddbrzjmc_encoder', 'QYSWDJXX_shdw_encoder',
       'QYSWDJXX_qy_encoder', 'QYSWDJXX_shjg_encoder', 'QYSWDJXX_djzclx_encoder',   
       'ZZDJBGXX_jzsx_deltaDay', 'zzsxcl_deltaDay', 'zzjzcl_deltaDay', 'zzsxhz_deltaDay', 'zzjzhz_deltaDay', 
       'jgdzzs_encoder', 'jyfw_encoder', 'sshymc_encoder',  
      ]
    
    #################################################################################
    # 数据增强
    temp_shixin_df = train_df[train_df.is_shixin == 1].copy()
    temp_notshixin_df = train_df[(train_df.is_shixin == 0)].copy()
    temp_train_df_list = list()
    skf_dataset = StratifiedKFold(n_splits=4, random_state=2018, shuffle=True)
    for index, (train_index, test_index) in enumerate(skf_dataset.split(temp_notshixin_df[shixin_fea], temp_notshixin_df['is_shixin'])):
        temp = temp_notshixin_df.iloc[train_index].copy()
        temp = pd.concat([temp, temp_shixin_df])
        temp.reset_index(inplace=True)
        temp_train_df_list.append(temp)
    
    #################################################################################
    # 失信任务的参数
    shixin_params = {
        'eta': 0.05,
        'max_depth': 6,
        'gamma': 1,
        'subsample': 0.9,
        'colsample_bytree': 0.95,
        'min_child_weight': 1,
        'max_delta_step': 1,
        'lambda': 30,
    }
    
    #################################################################################
    xgbModel_shixin = XgbModel(feaNames=shixin_fea, params=shixin_params)

    for i in range(4):
        temp_train_df = temp_train_df_list[i]
        test_df['FORTARGET1_round' + str(i)] = 0
        xgbModel_shixin.trainCV(temp_train_df[shixin_fea].values, temp_train_df['is_shixin'].values)
        test_df.loc[:, 'FORTARGET1_round' + str(i)] = xgbModel_shixin.predict(test_df[shixin_fea].values)

    #################################################################################
    test_df['FORTARGET1'] = test_df['FORTARGET1_round0'] * 0.25 + test_df['FORTARGET1_round1'] * 0.25 + test_df['FORTARGET1_round2'] * 0.25 + test_df['FORTARGET1_round3'] * 0.25

    #################################################################################
    print(np.mean(test_df['FORTARGET1']))

    def shixin_after_deal(df):
        FORTARGET1 = df['FORTARGET1']
        is_suzhou = df['is_suzhou']
        is_fr = df['is_fr']
        is_fzc = df['is_fzc']
        is_fz = df['is_fz']
        is_qq = df['is_qq']

        if (is_suzhou == 0) | (is_fr == 1):
            FORTARGET1 = 0
        if is_fzc == 1:
            FORTARGET1 = FORTARGET1 / 1.5
        if is_fz:
            FORTARGET1 = FORTARGET1 / 3
        if is_qq:
            FORTARGET1 = FORTARGET1 / 2

        return FORTARGET1

    test_df['FORTARGET1'] = test_df.apply(shixin_after_deal, axis=1)
    print(np.mean(test_df['FORTARGET1']))

    ##############################################################################
    ##############################################################################
    chufa_fea = ['zczj', 'fzjgsf', 'fzjgzt', 'fzhydm', 'fzjgqx', 'fzjglx', 'sjly', 'rwbh', 
       'QYFZCHRD_yynx', 'sshydm_encoder', 'xzgh', 'qyjd', 'qywd', 
       'QYBZRYXX_number', 'QYFZCHRD_number', 'QYSWDJXX_number', 'FRXZXKZX_number',
       'SDBDJQQJMCF_number', 'SGSFRXZXKXX_number', 'XKZZNJXX_number',
       'ZPSJ_number', 'ZZDJBGXX_number', 'QYJL_number', 
       'zczbbzmc_encoder', 'hymldm_encoder', 'zsszdsf_encoder', 'fzhyml_encoder',
       'xxtgbmmc_encoder', 'zl_encoder', 'sszj_encoder', 'fzjgmc_encoder',
       'qyjglxmc_encoder', 'qylxmc_encoder', 'QYFZCHRD_swglm_encoder',
       'QYFZCHRD_gljg_encoder', 'QYFZCHRD_djzclx_encoder', 
       'SGSFRXZXKXX_dfbm', 'SGSFRXZXKXX_xknr_encoder', 'SGSFRXZXKXX_xmmc_encoder', 'SGSFRXZXKXX_splb_encoder', 
       'SGSFRXZXKXX_xkjg_encoder', 'SGSFRXZXKXX_xysyfw_encoder', 'SGSFRXZXKXX_sjzt2_encoder', 
       'XKZZNJXX_njnd_encoder', 'XKZZNJXX_njjgmc_encoder', 'XKZZNJXX_njsxmc_encoder', 
       'ZZDJBGXX_zzmc_encoder', 'ZZDJBGXX_zyfw_encoder', 'ZZDJBGXX_rdjgqc_encoder', 'ZZDJBGXX_zzzsbh_encoder', 
       'QY_tzrgs', 'QY_zwzls', 'QY_fddbrbz_gs', 'QY_sxdbbz_gs',              
       'fzsw_deltaDay', 'clhz_deltaDay', 
       'clrq_month', 'clrq_day', 'hzrq_year', 'hzrq_month', 'hzrq_day', 'clrq_year',   
       'jgdzzs_encoder', 'jyfw_encoder', 'sshymc_encoder',
       'QYSWDJXX_fddbrzjmc_encoder', 'QYSWDJXX_shdw_encoder',
       'QYSWDJXX_qy_encoder', 'QYSWDJXX_shjg_encoder', 'QYSWDJXX_djzclx_encoder',
       'fzrq_encoder', 'bghzrq_encoder',
       'gszch_encoder', 'fddbrzjhm_encoder', 'zzjgdm_encoder',     
       'jyfw_array_0_fea', 'jyfw_array_1_fea', 'jyfw_array_2_fea',
       'jyfw_array_3_fea', 'jyfw_array_4_fea', 'jyfw_array_5_fea',
       'jyfw_array_6_fea', 'jyfw_array_7_fea', 'jyfw_array_8_fea',
       'jyfw_array_9_fea',
      ]

    ##############################################################################
    chufa_params = {
        'eta': 0.03,
        'max_depth': 13,
        'gamma': 1,
        'subsample': 0.9,
        'colsample_bytree': 0.95,
        'min_child_weight': 3,
        'max_delta_step': 1,
        'lambda': 30,
    }

    ##############################################################################
    xgbModel_chufa = XgbModel(feaNames=chufa_fea, params=chufa_params)

    # 线下模型
    startTime = datetime.now()
    xgbModel_chufa.trainCV(train_df[chufa_fea].values, train_df['is_chufa'].values)
    xgbModel_chufa.getFeaScore(show=True)
    print('training time: ', datetime.now()-startTime)

    ##############################################################################
    test_df.loc[:,'FORTARGET2'] = xgbModel_chufa.predict(test_df[chufa_fea].values)
    print(np.mean(test_df['FORTARGET2']))
    
    ##############################################################################
    test_df.rename(columns={'qymc':'EID'}, inplace=True)
    print(test_df[['EID', 'FORTARGET1', 'FORTARGET2']].head())

    ##############################################################################
    exportResult(test_df[['EID', 'FORTARGET1', 'FORTARGET2']], 'xgb_online')