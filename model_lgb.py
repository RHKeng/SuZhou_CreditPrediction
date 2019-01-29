# -*- coding: utf-8 -*-

import datetime
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# 导出预测结果
def exportResult(df, fileName):
    df.to_csv('../result/%s.csv' % fileName, header=True, index=False)

if __name__ == "__main__":
    ###################################################################
    # 导入数据
    print('导入数据！')
    train_df = pd.read_csv('../temp/fusai_train_df_all.csv')
    test_df = pd.read_csv('../temp/fusai_test_df_all.csv')
    
    ###################################################################
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
    print(len(after_test_df))
    suzhou_set = set(after_test_df[after_test_df.zsszdsf == '江苏省']['qymc'])
    print(len(suzhou_set))
    test_df['is_suzhou'] = test_df['qymc'].map(lambda x : 1 if x in suzhou_set else 0)
    print(len(test_df[test_df.is_suzhou == 1]))
    print(len(test_df))
    
    # 类别特征
    categorical_feature_shixin = ['zsszdsf_encoder', 'qyjglxmc_encoder', 'hymldm_encoder', 
                                 'zczbbzmc_encoder', 'fzjglx', 'fzhydm', 'fzhyml_encoder', 'fzjgzt',
                                 'fzjgsf', 'qylxmc_encoder', 'xzgh', 'sshydm_encoder', 'zl_encoder', 'fzjgmc_encoder',]

    categorical_feature_chufa = ['zsszdsf_encoder', 'qyjglxmc_encoder', 'hymldm_encoder', 
                                'zczbbzmc_encoder', 'fzjglx', 'fzhydm', 'fzhyml_encoder', 'fzjgzt',
                                'fzjgsf', 'qylxmc_encoder', 'xzgh', 'sshydm_encoder', 'zl_encoder',  'fzjgmc_encoder',]

    ###################################################################
    FRXZXKZX_test_df = pd.read_csv('../data/fusai/test/法人行政许可注（撤、吊）销信息.csv')
    fr_qymc_set = set(FRXZXKZX_test_df['企业名称'])

    test_df['is_fr'] = test_df['qymc'].map(lambda x : 1 if x in fr_qymc_set else 0)

    QYFZCHRD_test_df = pd.read_csv('../data/fusai/test/企业非正常户认定.csv')
    fzc_qymc_set = set(QYFZCHRD_test_df['企业名称'])

    test_df['is_fzc'] = test_df['qymc'].map(lambda x : 1 if x in fzc_qymc_set else 0)

    ###################################################################
    FZJGXX_test_df = pd.read_csv('../data/fusai/test/分支机构信息.csv')
    fz_qymc_set = set(FZJGXX_test_df['企业名称'])

    test_df['is_fz'] = test_df['qymc'].map(lambda x : 1 if x in fz_qymc_set else 0)

    ###################################################################
    SDBDJQQJMCF_test_df = pd.read_csv('../data/fusai/test/双打办打击侵权假冒处罚案件信息.csv')
    qq_qymc_set = set(SDBDJQQJMCF_test_df['企业名称'])

    test_df['is_qq'] = test_df['qymc'].map(lambda x : 1 if x in qq_qymc_set else 0)

    
    ###################################################################
    train_df['shixin_weight'] = 1
    train_df.loc[train_df.is_chufa == 1, 'shixin_weight'] = 2

    ###################################################################
    shixin_fea = ['zczj', 'fzjgsf', 'fzjgzt', 'fzhydm', 'fzjgqx', 'fzjglx', 'sjly', 'rwbh', 
       'QYFZCHRD_yynx', 'sshydm_encoder', 'xzgh', 'qyjd', 'qywd', 
       'zczbbzmc_encoder', 'hymldm_encoder', 'zsszdsf_encoder', 'fzhyml_encoder',
       'xxtgbmmc_encoder', 'zl_encoder', 'sszj_encoder', 'fzjgmc_encoder',
       'qyjglxmc_encoder', 'qylxmc_encoder',
       'fzsw_deltaDay', 'clhz_deltaDay', 
       'clrq_month', 'clrq_day', 'hzrq_year', 'hzrq_month', 'hzrq_day', 'clrq_year',              
       'ZPSJ_zprs_max', 'ZPSJ_zprs_min', 'ZPSJ_zprs_mean', 'ZPSJ_zprs_std',       
       'ZZDJBGXX_jzsx_deltaDay', 'zzsxcl_deltaDay', 'zzjzcl_deltaDay', 'zzsxhz_deltaDay', 'zzjzhz_deltaDay',             
       'jgdzzs_encoder', 'jyfw_encoder', 'sshymc_encoder',
       'XKZZNJXX_xxtgbmmc_encoder', 'QYFZCHRD_zcdz_encoder', 
       'QYFZCHRD_rwbh_encoder', 'QYFZCHRD_xxtgbmmc_encoder',         
       'jgdzzs_top3_encoder', 'jgdzzs_top6_encoder',
      ]

    ###################################################################
    temp_shixin_df = train_df[train_df.is_shixin == 1].copy()
    temp_notshixin_df = train_df[(train_df.is_shixin == 0)].copy()
    temp_train_df_list = list()
    skf_dataset = StratifiedKFold(n_splits=4, random_state=0, shuffle=True)
    for index, (train_index, test_index) in enumerate(skf_dataset.split(temp_notshixin_df[shixin_fea], temp_notshixin_df['is_shixin'])):
        temp = temp_notshixin_df.iloc[test_index].copy()
        temp = pd.concat([temp, temp_shixin_df])
        temp.reset_index(inplace=True)
        temp_train_df_list.append(temp)

    ###################################################################
    lgb_model_shixin = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=127, reg_alpha=6, reg_lambda=10, max_bin=425,
    max_depth=-1, n_estimators=5000, objective='binary',
    colsample_bytree=1, subsample_freq=1,
    learning_rate=0.02, random_state=2018, n_jobs=-1,min_child_samples=5
    )

    skf = StratifiedKFold(n_splits=10, random_state=2018, shuffle=True)
    for i in range(4):
        best_score = []
        temp_train_df = temp_train_df_list[i]
        test_df['FORTARGET1_round' + str(i)] = 0
        number = 0
        for index, (train_index, test_index) in enumerate(skf.split(temp_train_df[shixin_fea], temp_train_df['is_shixin'])):
            weight = temp_train_df['shixin_weight'].iloc[train_index].tolist()
            lgb_model_shixin.fit(temp_train_df[shixin_fea].iloc[train_index], temp_train_df['is_shixin'][train_index],
                             sample_weight=weight,
                             categorical_feature=categorical_feature_shixin,
                      eval_set=[(temp_train_df[shixin_fea].iloc[train_index], temp_train_df['is_shixin'][train_index]),
                                (temp_train_df[shixin_fea].iloc[test_index], temp_train_df['is_shixin'][test_index])], early_stopping_rounds=200, eval_metric='auc')
            if lgb_model_shixin.best_score_['valid_1']['auc'] > 0.9:
                best_score.append(lgb_model_shixin.best_score_['valid_1']['auc'])
                print(best_score)
                test_pred = lgb_model_shixin.predict_proba(test_df[shixin_fea], num_iteration=lgb_model_shixin.best_iteration_)[:, 1]
                print('test mean:', test_pred.mean())
                test_df['FORTARGET1_round' + str(i)] = test_df['FORTARGET1_round' + str(i)] + test_pred
                number = number + 1
        print(number)
        print(np.mean(best_score))
        test_df['FORTARGET1_round' + str(i)] = test_df['FORTARGET1_round' + str(i)] / number

    ###################################################################
    test_df['FORTARGET1'] = test_df['FORTARGET1_round0'] * 0.25 + test_df['FORTARGET1_round1'] * 0.25 + test_df['FORTARGET1_round2'] * 0.25 + test_df['FORTARGET1_round3'] * 0.25

    ###################################################################
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
    
    ###################################################################
    ###################################################################
    train_df['chufa_weight'] = 1
    train_df.loc[train_df.is_chufa == 1, 'chufa_weight'] = 2

    ###################################################################
    chufa_fea = ['zczj', 'fzjgsf', 'fzjgzt', 'fzhydm', 'fzjgqx', 'fzjglx', 'sjly', 'rwbh', 
       'QYFZCHRD_yynx', 'sshydm_encoder', 'xzgh', 'qyjd', 'qywd', 
       'QYBZRYXX_number', 'QYFZCHRD_number', 'QYSWDJXX_number', 'FRXZXKZX_number',
       'SDBDJQQJMCF_number', 'SGSFRXZXKXX_number', 'XKZZNJXX_number',
       'ZPSJ_number', 'ZZDJBGXX_number', 'QYJL_number',      
       'zczbbzmc_encoder', 'hymldm_encoder', 'zsszdsf_encoder', 'fzhyml_encoder',
       'xxtgbmmc_encoder', 'zl_encoder', 'sszj_encoder', 'fzjgmc_encoder',
       'qyjglxmc_encoder', 'qylxmc_encoder',  
       'QYFZCHRD_swglm_encoder',
       'QYFZCHRD_gljg_encoder', 'QYFZCHRD_djzclx_encoder', 
       'SGSFRXZXKXX_dfbm', 'SGSFRXZXKXX_xknr_encoder', 'SGSFRXZXKXX_xmmc_encoder', 'SGSFRXZXKXX_splb_encoder', 
       'SGSFRXZXKXX_xkjg_encoder', 'SGSFRXZXKXX_xysyfw_encoder', 'SGSFRXZXKXX_sjzt2_encoder', 
       'XKZZNJXX_njnd_encoder', 'XKZZNJXX_njjgmc_encoder', 'XKZZNJXX_njsxmc_encoder', 
       'ZZDJBGXX_zzmc_encoder', 'ZZDJBGXX_zyfw_encoder', 'ZZDJBGXX_rdjgqc_encoder', 'ZZDJBGXX_zzzsbh_encoder', 
       'QY_tzrgs', 'QY_zwzls', 'QY_fddbrbz_gs', 'QY_sxdbbz_gs',               
       'fzsw_deltaDay', 'clhz_deltaDay', 
       'clrq_month', 'clrq_day', 'hzrq_year', 'hzrq_month', 'hzrq_day', 'clrq_year',
       'jgdzzs_top3_encoder', 'jgdzzs_top6_encoder',
       'QYSWDJXX_fddbrzjmc_encoder', 'QYSWDJXX_shdw_encoder',
       'QYSWDJXX_qy_encoder', 'QYSWDJXX_shjg_encoder', 'QYSWDJXX_djzclx_encoder',    
       'jgdzzs_encoder', 'jyfw_encoder', 'sshymc_encoder',
       'jyfw_array_0_fea', 'jyfw_array_1_fea', 'jyfw_array_2_fea',
       'jyfw_array_3_fea', 'jyfw_array_4_fea', 'jyfw_array_5_fea',
       'jyfw_array_6_fea', 'jyfw_array_7_fea', 'jyfw_array_8_fea',
       'jyfw_array_9_fea',
       'fzrq_encoder', 'bghzrq_encoder',
       'gszch_encoder', 'fddbrzjhm_encoder', 'zzjgdm_encoder', 
      ]

    ###################################################################
    lgb_model_chufa = lgb.LGBMClassifier(
    boosting_type='gbdt', num_leaves=127, reg_alpha=6, reg_lambda=10, max_bin=425,
    max_depth=-1, n_estimators=5000, objective='binary',
#     subsample=0.8, 
    colsample_bytree=1, subsample_freq=1,
    learning_rate=0.05, random_state=2018, n_jobs=-1,min_child_samples=5
    )

    test_df['FORTARGET2'] = 0

    skf = StratifiedKFold(n_splits=10, random_state=2018, shuffle=True)
    best_score = []
    for index, (train_index, test_index) in enumerate(skf.split(train_df[chufa_fea], train_df['is_chufa'])):
        weight = train_df['chufa_weight'].iloc[train_index].tolist()
        lgb_model_chufa.fit(train_df[chufa_fea].iloc[train_index], train_df['is_chufa'][train_index],
                        sample_weight=weight,
                        categorical_feature=categorical_feature_chufa,
                        eval_set=[(train_df[chufa_fea].iloc[train_index], train_df['is_chufa'][train_index]),
                            (train_df[chufa_fea].iloc[test_index], train_df['is_chufa'][test_index])], early_stopping_rounds=100, eval_metric='auc')
        best_score.append(lgb_model_chufa.best_score_['valid_1']['auc'])
        print(best_score)
        test_pred = lgb_model_chufa.predict_proba(test_df[chufa_fea], num_iteration=lgb_model_shixin.best_iteration_)[:, 1]
        print('test mean:', test_pred.mean())
        test_df['FORTARGET2'] = test_df['FORTARGET2'] + test_pred
    print(np.mean(best_score))

    test_df['FORTARGET2'] = test_df['FORTARGET2'] / 10
    mean = test_df['FORTARGET2'].mean()
    print('mean:', mean)

    ###################################################################
    xgb_online = pd.read_csv('../temp/xgb_online.csv')

    test_df.rename(columns={'FORTARGET1':'lgb_FORTARGET1', 'FORTARGET2':'lgb_FORTARGET2'}, inplace=True)

    test_df['xgb_FORTARGET1'] = xgb_online['FORTARGET1']
    test_df['xgb_FORTARGET2'] = xgb_online['FORTARGET2']

    print(test_df[['lgb_FORTARGET1', 'lgb_FORTARGET2', 'xgb_FORTARGET1', 'xgb_FORTARGET2']].head())
    
    ###################################################################
    test_df['FORTARGET1'] = test_df['xgb_FORTARGET1'] * 0.6 + test_df['lgb_FORTARGET1'] * 0.4

    test_df['FORTARGET2'] = test_df['xgb_FORTARGET2'] * 0.2 + test_df['lgb_FORTARGET2'] * 0.8

    ###################################################################
    test_df.rename(columns={'qymc':'EID'}, inplace=True)
    print(test_df[['EID', 'FORTARGET1', 'FORTARGET2']].head())

    ###################################################################
    exportResult(test_df[['EID', 'FORTARGET1', 'FORTARGET2']], 'compliance_assessment')
