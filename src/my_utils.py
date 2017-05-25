#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# @author edvard
# @date 2017-05-19

import pandas as pd
import pickle

def my_read_csv(path):
    '''
    根据path，读取 csv 文件，这里使用了 chunk,可以避免 OOM
    :param path: 
    :return: 
    '''
    reader = pd.read_csv(path, iterator=True)
    loop = True
    chunkSize = 100000
    chunks = []
    while loop:
        try:
            chunk = reader.get_chunk(chunkSize)
            chunks.append(chunk)
        except:
            loop = False
            print path + " 已经加载."

    return pd.concat(chunks)

def read_sample_data(path="../sample_data/"):
    train = pickle.load(open(path + "train"))
    installedapps = pickle.load(open(path + "installedapps"))
    actions = pickle.load(open(path + "actions"))
    users = pickle.load(open(path + "users"))
    return train, installedapps, actions, users
