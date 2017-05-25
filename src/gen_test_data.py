#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @author edvard
# @date 2017-05-19

import pandas as pd
import numpy as np
from gen_train_data import get_user_app_preference_df, age_convert
from my_utils import *
from my_constant import *


if __name__ == "__main__":
    # 加载原始的测试集
    test = my_read_csv(TEST_DATA_PATH)
    train = my_read_csv(TRAIN_DATA_PATH)
    installedapps = my_read_csv(INSTALLED_APPS_DATA_PATH)
    actions = my_read_csv(ACTION_DATA_PATH)
    users = my_read_csv(USER_DATA_PATH)

    # 获取测试数据的相关信息
    test_users_ids = test.userID.unique()
    installedapps = installedapps[installedapps.userID.isin(test_users_ids)]
    actions = actions[actions.userID.isin(test_users_ids)]
    users = users[users.userID.isin(test_users_ids)]

    # 用户相关特征
    users.age = users.age.map(age_convert)
    age_df = pd.get_dummies(users.age, prefix="age")
    gender_df = pd.get_dummies(users.gender, prefix="sex")
    education_df = pd.get_dummies(users.education, prefix='education')
    marriage_df = pd.get_dummies(users.marriageStatus, prefix='marriage')
    baby_df = pd.get_dummies(users.haveBaby, prefix='baby')
    users = pd.concat([users, age_df, gender_df, education_df, marriage_df, baby_df], axis=1)
    users = users.drop(['age', 'gender', 'education', 'marriageStatus', 'haveBaby'], axis=1)

    # 获取 train, installedapps, actions 中全部的 app_category
    all_app_categories = np.unique(np.append(test.appCategory.unique(), [installedapps.appCategory.unique(), actions.appCategory.unique()]))
    all_app_categories = ["app_category_" + str(app_category) for app_category in all_app_categories]
    # 获取 ad 里面的应用分类
    all_ad_app_categories = test.appCategory.unique()
    all_ad_app_categories = ["ad_app_category_" + str(ad_app_category) for ad_app_category in all_ad_app_categories]

    # 应用转化率的特征，对于为空的，填充为训练集里面的平均值
    key = "appID"
    dfCvr = train.groupby(key).apply(lambda df: np.mean(df["label"])).reset_index()
    dfCvr.columns = [key, "avg_cvr"]
    test = pd.merge(test, dfCvr, how="left", on=key)
    test["avg_cvr"].fillna(np.mean(train["label"]), inplace=True)

    # one hot 应用平台
    platform_df = pd.get_dummies(test.appPlatform, prefix="platform")
    test = pd.concat([test, platform_df], axis=1)
    test = test.drop(['appPlatform'], axis=1)

    # 因为第 30 天的数据存在不准确的问题，所以这里只取前 29 天的数据
    # TODO 后期需要想一下如何处理第 30 天数据不准确的问题
    df = get_user_app_preference_df(29, installedapps, actions)

    df = pd.merge(test, df, on=['userID'], how='left')
    df = pd.merge(df, users, on=['userID'], how='left')
    # 广告中应用类别特征
    ad_app_category = pd.get_dummies(df.appCategory, prefix="ad_app_category")
    df = pd.concat(
        [df, ad_app_category],
        axis=1
    )
    # 暂时去除掉不需要的字段
    # TODO 回流时间这个特征需要好好琢磨一下
    df = df.drop(
        ['userID', 'date', 'hour', 'time', 'clickTime', 'adID', 'advertiserID', 'camgaignID', 'appID',
         'appCategory'], axis=1)
    # 检查矩阵是否缺少某个类别，并将他补上
    for app_category in all_app_categories:
        if app_category not in df:
            df[app_category] = 0
    for ad_app_category in all_ad_app_categories:
        if ad_app_category not in df:
            df[ad_app_category] = 0
    lack_app_categories = ['ad_app_category_203', 'ad_app_category_408', 'ad_app_category_0', 'ad_app_category_106']
    for lack_app_category in lack_app_categories:
        if lack_app_category not in df:
            df[lack_app_category] = 0

    # 存在缺失值的，都填充上0
    df.fillna(0)

    # 储存测试集的数据
    pickle.dump([df], open("../cache/test_input_data", 'w'))

    print "测试输入数据已经生成"
