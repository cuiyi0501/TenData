#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @author edvard
# @date 2017-05-19

import numpy as np
import pickle

from functools import reduce
from my_constant import *
from my_utils import *

def filter_by_intersect_users(minibatch=False):
    train = my_read_csv(TRAIN_DATA_PATH)
    installedapps = my_read_csv(INSTALLED_APPS_DATA_PATH)
    actions = my_read_csv(ACTION_DATA_PATH)
    users = my_read_csv(USER_DATA_PATH)

    # 取 train & installedapps & action 中都有数据的用户
    train_user_ids = train.userID.unique()
    installedapps_user_ids = installedapps.userID.unique()
    actions_user_ids = actions.userID.unique()

    intersect_user_ids = reduce(np.intersect1d, (train_user_ids, installedapps_user_ids, actions_user_ids))

    # 判断是不是随机抽取一小部分数据，
    # 如果是则随机抽取 50000 个用户的 user_ids
    if minibatch:
         sample_user_ids = intersect_user_ids[np.random.randint(len(intersect_user_ids), size=50000)]
    else:
        sample_user_ids = intersect_user_ids

    sample_train = train[train.userID.isin(sample_user_ids)]
    sample_installedapps = installedapps[installedapps.userID.isin(sample_user_ids)]
    sample_actions = actions[actions.userID.isin(sample_user_ids)]
    sample_user = users[users.userID.isin(sample_user_ids)]
    pickle.dump(sample_train, open("../sample_data/train", 'w'))
    pickle.dump(sample_installedapps, open("../sample_data/installedapps", 'w'))
    pickle.dump(sample_actions, open("../sample_data/actions", 'w'))
    pickle.dump(sample_user, open("../sample_data/users", 'w'))
    print "用户人数：" + str(len(sample_user_ids))
    print "采样完成，用户来自于 train & installedapps & action "
    return sample_user, sample_installedapps, sample_actions, sample_train

def filter_by_train_users(minibatch=False):
    train = my_read_csv(TRAIN_DATA_PATH)
    installedapps = my_read_csv(INSTALLED_APPS_DATA_PATH)
    actions = my_read_csv(ACTION_DATA_PATH)
    users = my_read_csv(USER_DATA_PATH)

    # 取 train 里面所有的用户 ids
    train_user_ids = train.userID.unique()
    installedapps_user_ids = installedapps.userID.unique()
    intersect_user_ids = np.intersect1d(train_user_ids, installedapps_user_ids, assume_unique=True)

    if minibatch:
        train_user_ids = train_user_ids[np.random.randint(len(train_user_ids), size=20000)]

    train = train[train.userID.isin(train_user_ids)]
    installedapps = installedapps[installedapps.userID.isin(train_user_ids)]
    actions = actions[actions.userID.isin(train_user_ids)]
    train_users = users[users.userID.isin(train_user_ids)]

    pickle.dump(train, open("../sample_data/filter_by_train_user/train", 'w'))
    pickle.dump(installedapps, open("../sample_data/filter_by_train_user/installedapps", 'w'))
    pickle.dump(actions, open("../sample_data/filter_by_train_user/actions", 'w'))
    pickle.dump(train_users, open("../sample_data/filter_by_train_user/users", 'w'))

    intersect_user_ids = np.intersect1d(intersect_user_ids, train_user_ids, assume_unique=True)
    print "其中 " + str(len(train_user_ids) - len(intersect_user_ids)) + " 是没有历史安装记录信息的"
    print "用户人数：" + str(len(train_user_ids))
    print "采样完成，用户来自于 train"
    return train_users, installedapps, actions, train



if __name__ == "__main__":
    users, installedapps, actions, train = filter_by_train_users(minibatch=False)