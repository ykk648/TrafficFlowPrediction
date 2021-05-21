#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

X = np.load('E:/PycharmProjects/TDRL/predict_data.npy')
Y = np.load('E:/PycharmProjects/TDRL/data_single_lane.npy')
print(Y)
print(X.shape)
print(Y.shape)

data = X[0]

print(data.shape)

data_pd = DataFrame(data, columns=['traffic flow'],
                    index=pd.date_range('2018-10-08 00:05:00', periods=6048, freq='300s'))

data_1day = data_pd[0:288]

# print(data_1day)

# data_1day.plot()
# plt.show()
# # 绘制自相关图
# plot_acf(data_1day).show()
# # 绘制偏自相关图
plot_pacf(data_1day).show()

from statsmodels.tsa.stattools import adfuller as ADF

adf = ADF(data_1day['traffic flow'])
print(adf)

D_data = data_pd.diff().dropna()
D_data.columns = ['traffic flow']
# D_data.plot()
# plt.show()

adf = ADF(D_data['traffic flow'])
print(adf)
# plot_acf(D_data).show()

# 绘制自相关图
plot_acf(D_data).show()
# 绘制偏自相关图
plot_pacf(D_data).show()
