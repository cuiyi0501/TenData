#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @author edvard
# @date 2017-05-19

import pickle
import xgboost as xgb
import scipy as sp
import pandas as pd
import numpy as np


def get_model_paramters():
    param = {
        'n_estimators': 1000,
        'max_depth': 5,
        'min_child_weight': 5,
        'gamma': 0,
        'subsample': 0.6,
        'colsample_bytree': 0.8,
        'scale_pos_weight': 1,
        'eta': 0.05,
        'silent': 1,
        'objective': 'binary:logistic'
    }

    # 136 181
    num_round = 181
    param['nthread'] = 5
    param['eval_metric'] = "logloss"
    return param.items(), num_round

def xgb_cv(data, simple=False):
    # 交叉验证，因为是时间序列的关系，所以交叉验证方式略有不同
    # 前 1~17 天的数据训练，预测第 18 天
    # 前 1~18 天的数据训练，预测第 19 天
    # ... 以此类推，这样划分的话则共进行了 13-flod validation

    # 获取模型参数
    plst, num_round = get_model_paramters()

    # 保存每次 k-flod 的结果
    evals_results = []
    train_df = data[0]
    start = 1

    # 只验证最后两轮
    if simple:
        start = 11

    # 最后那天的数据有问题，这里去除掉
    # TODO 后期想一下如何优化最后一天的数据不准确的问题
    for i in range(start, len(data)-1):
        validation_df = data[i]
        for j in range(1, i):
            train_df.append(data[j], ignore_index=True)
        print str(i) + "-fold 开始....."
        train_label = train_df.label
        tmp_train_df = train_df.copy().drop(['label'], axis=1)
        test_label = validation_df.label
        validation_df = validation_df.drop(['label'], axis=1)

        dtrain=xgb.DMatrix(tmp_train_df.values, label=train_label)
        dvalidation=xgb.DMatrix(validation_df.values, label=test_label)
        evallist = [(dvalidation, 'eval'), (dtrain, 'train')]
        evals = {}
        # 训练模型
        xgb.train(plst, dtrain, num_round, evallist, evals_result=evals)
        evals_results.append(evals)

    mean_evals = []
    for i in range(0, len(evals_results)):
        eval = evals_results[i]
        print "第" + str(i) + "训练"
        print "训练集误差 " + str(eval['train']['logloss'][-1])
        print "验证集误差 " + str(eval['eval']['logloss'][-1])
        mean_evals.append(eval['eval']['logloss'][-1])
    print "验证集平均误差" + str(np.mean(mean_evals))

def submission(train, test):
    plst, num_round = get_model_paramters()
    # 取 train 的最后两段数据做训练集和验证集
    validation_df = train[12]

    train_df = train[0]
    for j in range(1, 12):
        train_df.append(train[j], ignore_index=True)
    train_label = train_df.label
    train_df = train_df.drop(['label'], axis=1)
    test_label = validation_df.label
    validation_df = validation_df.drop(['label'], axis=1)
    dtrain = xgb.DMatrix(train_df.values, label=train_label)
    dvalidation = xgb.DMatrix(validation_df.values, label=test_label)
    evallist = [(dvalidation, 'eval'), (dtrain, 'train')]
    eval = {}
    bst = xgb.train(plst, dtrain, num_round, evallist, evals_result=eval)

    # diff = np.subtract(eval['train']['logloss'], eval['eval']['logloss'])
    # print "验证和训练误差最小的迭代次数为" + str(np.argmin(diff) + 1)

    test = test[0]
    test = test.drop(['label', 'instanceID'], axis=1)

    res = pd.DataFrame({'instanceID': range(1, test.shape[0] + 1)})
    dtest = xgb.DMatrix(test.values)
    res['proba'] = bst.predict(dtest)
    res.to_csv("../res/submission.csv", index=False)
    # with zipfile.ZipFile("submission.zip", "w") as fout:
    #     fout.write("../res/submission.csv", compress_type=zipfile.ZIP_DEFLATED)


def logloss(act, pred):
  epsilon = 1e-15
  pred = sp.maximum(epsilon, pred)
  pred = sp.minimum(1-epsilon, pred)
  ll = sum(act*sp.log(pred) + sp.subtract(1,act)*sp.log(sp.subtract(1,pred)))
  ll = ll * -1.0/len(act)
  return ll

if __name__ == "__main__":
    # 加载训练数据
    train = pickle.load(open('../cache/train_input_data'))
    test = pickle.load(open("../cache/test_input_data"))

    print "Train 特征总数", [elem.shape[1] for elem in train]

    # print "Test 特征总数", [elem.shape[1] for elem in test]
    # 查看 train 和 test 之间是否存在不同的地方
    # print set(list(train[0])).symmetric_difference(set(list(test[0])))

    # print "训练数据正样本所占比例: ", [round(elem.label.value_counts()[1]/float(elem.shape[0]), 4) for elem in train]


    # 线下 CV
    # xgb_cv(train, simple=True)

    # 预测
    submission(train, test)
