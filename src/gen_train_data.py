# -*- coding:utf-8 -*-
# @author edvard
# @date 2017-05-20

import pandas as pd
import numpy as np
from my_utils import read_sample_data
import pickle

def get_user_app_preference_df(days, installedapps, actions):
    '''
    获得 用户-应用类型 偏好矩阵
    :param days: 
    :param installedapps: 
    :param actions: 
    :return: 
    '''
    app_category = pd.get_dummies(installedapps.appCategory, prefix="app_category")
    installedapps = pd.concat(
        [installedapps, app_category],
        axis=1
    )
    installedapps = installedapps.fillna(0)
    # 删除掉 appCategory 和 appID 这个字段
    installedapps = installedapps.drop(['appID', 'appCategory'], axis=1)
    # 同样的，将 actions 转化成偏好矩阵
    actions = actions[actions.date <= days]
    action_app_category = pd.get_dummies(actions.appCategory, prefix="app_category")
    actions = pd.concat(
        [actions, action_app_category],
        axis=1
    )
    actions = actions.fillna(0)
    actions = actions.drop(['date', 'hour', 'minutes', 'appID', 'installTime', 'appCategory'], axis=1)
    # 拼接在一起
    installedapps = installedapps.append(actions, ignore_index=True)
    # 加上用户 ID 后返回
    installedapps_df = installedapps.groupby("userID").sum()

    # 应用流行度和最大值
    app_category_popularity = installedapps_df.sum(axis=0)
    # 获得逆流行度
    inverse_popularity = np.log10(max(app_category_popularity)/app_category_popularity)

    # 求和每行的值
    each_row_sums = installedapps_df.sum(axis=1)
    # 再除于每行，获得每个应用的占比
    installedapps_df = installedapps_df.div(each_row_sums, axis=0)
    # 乘于逆流行度
    installedapps_df = installedapps_df.multiply(inverse_popularity, axis=1)

    installedapps_df['userID'] = installedapps_df.index.values
    return installedapps_df

def age_convert(age):
    '''
    转换用户年龄的表示
    :param age: 
    :return: 
    '''
    if age == 0:
        return 0
    elif 0 < age <= 15:
        return 1
    elif 15 < age <= 25:
        return 2
    elif 25 < age <= 35:
        return 3
    elif 35 < age <= 45:
        return 4
    elif 45 < age <= 55:
        return 5
    elif 55 < age:
        return 6
    else:
        return -1

if __name__ == "__main__":
    # 取得数据
    # train, installedapps, actions, users = read_sample_data(path="../sample_data/")
    train, installedapps, actions, users = read_sample_data(path="../sample_data/filter_by_train_user/")

    users.age = users.age.map(age_convert)
    age_df = pd.get_dummies(users.age, prefix="age")
    gender_df = pd.get_dummies(users.gender, prefix="sex")
    education_df = pd.get_dummies(users.education, prefix='education')
    marriage_df = pd.get_dummies(users.marriageStatus, prefix='marriage')
    baby_df = pd.get_dummies(users.haveBaby, prefix='baby')
    users = pd.concat([users, age_df, gender_df, education_df, marriage_df, baby_df], axis=1)
    users = users.drop(['age', 'gender', 'education', 'marriageStatus', 'haveBaby'], axis=1)

    # 获取 train, installedapps, actions 中全部的 app_category
    all_app_categories = np.unique(reduce(np.append, [train.appCategory.unique(), installedapps.appCategory.unique(), actions.appCategory.unique()]))
    all_app_categories = ["app_category_" + str(app_category) for app_category in all_app_categories]
    # 获取 ad 里面的应用分类
    all_ad_app_categories = train.appCategory.unique()
    all_ad_app_categories = ["ad_app_category_" + str(ad_app_category) for ad_app_category in all_ad_app_categories]

    # 特定应用转化率的特征
    key = "appID"
    dfCvr = train.groupby(key).apply(lambda df: np.mean(df["label"])).reset_index()
    dfCvr.columns = [key, "avg_cvr"]
    train = pd.merge(train, dfCvr, how="left", on=key)

    # 将 train 数据按照天数进行划分
    days = train.date.unique()
    # 储存 17 天到 30 天的输入数据
    input_chunks = []

    # 广告中应用类别特征
    feats = ["appCategory", "creativeID", "adID", "camgaignID", "advertiserID", "appPlatform"]
    for feat in feats:
        tmp = pd.get_dummies(train[feat], prefix=("ad_" + feat))
        train = pd.concat(
            [train, tmp],
            axis=1
        )
    train = train.drop(feats, axis=1)

    for day in days:
        # 获得第 day 天的 train 数据
        tmpTrain = train[train.date == day]
        # 获得 17~30(day 变量) 天前的用户-应用类别偏好矩阵
        df = get_user_app_preference_df(day-1, installedapps, actions)
        # 预测第 17 天的应用转化率的数据
        df = pd.merge(tmpTrain, df, on=['userID'], how='left')
        df = pd.merge(df, users, on=['userID'], how='left')

        # 暂时去除掉不需要的字段
        df = df.drop(['userID', 'date', 'hour', 'time', 'clickTime', 'conversionTime', 'appID'], axis=1)
        # 检查矩阵是否缺少某个类别，并将他补上
        for app_category in all_app_categories:
            if app_category not in df:
                df[app_category] = 0
        for ad_app_category in all_ad_app_categories:
            if ad_app_category not in df:
                df[ad_app_category] = 0
        # 存在缺失值的，都填充上0
        df.fillna(0)
        input_chunks.append(df)

    pickle.dump(input_chunks, open("../cache/train_input_data", 'w'))

    print "样本人数：" + str(users.shape[0])
    print "训练模型数据已经生成"






