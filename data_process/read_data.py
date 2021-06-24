#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

data = np.load('data.npy')
print(data)
print(data.shape)

# data2 = np.fromfile('data.csv',dtype=np.int).reshape(26,2,2016)
# print(data2)
# print(data2.shape)