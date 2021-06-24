#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn import datasets
# X, y = datasets.make_blobs(n_samples=500, n_features=6, centers=5, cluster_std=[0.4, 0.3, 0.4, 0.3, 0.4], random_state=11)
# print(X.shape) # (500,6)

# X = np.load('data.npy')
# X = X.reshape(26,4032)

X = np.load('data_single_lane.npy')
print(X.shape)

from sklearn.cluster import SpectralClustering
from sklearn import metrics

from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn import svm
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 离群点检测
# XX = np.load('data_single_lane-有离群点.npy')
# print(XX.shape)
# outliers_fraction = 0.1
# iof = IsolationForest(behaviour="new",contamination=outliers_fraction,random_state=42)
# y_p = iof.fit_predict(XX)
# svmm = svm.OneClassSVM(nu=outliers_fraction, kernel="poly",gamma='auto'
#                                  )
# y_pp = svmm.fit_predict(XX)
# print(y_p,y_pp)


# # 聚类
# n_cluster = 10
# y_pred = SpectralClustering(affinity='nearest_neighbors',n_clusters=n_cluster, n_neighbors=11).fit_predict(X)  # 更稳定
# #
# #
# print(type(y_pred))
# print ("Calinski-Harabasz Score", metrics.calinski_harabaz_score(X, y_pred) )
# print(y_pred)
#
# predict_data = []
# for i in range(n_cluster):
#     # print(i)
#     index_list = np.where(y_pred == i)
#     print(index_list)
#     data_temp = np.zeros(6048)
#
#     # print(X[index_list[0][0]])
#     # print(X[index_list[0]])
#     #
#     # scaler = MinMaxScaler(feature_range=(0, 1)).fit(X[index_list[0][0]].reshape(-1, 1))
#
#     for j in range(len(index_list[0])):
#         # print(index_list[j])
#         # temp = index_list[j]
#         # print(temp)
#         X_list = X[index_list[0][j]]
#         # print(X_list.shape)
#         # for ii in range(len(X_list)):
#         #     data_after_cluster[ii]+=X_list[ii]
#         # print(X_list)
#         # X_list = scaler.transform(X_list.reshape(-1, 1)).reshape(1, -1)[0]
#         cc = data_temp + X_list//len(index_list[0])
#         data_temp = cc
#     # data_temp = scaler.inverse_transform(data_temp.reshape(-1,1)).reshape(1, -1)[0]
#     # data_temp = data_temp//len(index_list[0])
#     # print(data_temp)
#     # print(data_temp.shape)
#     predict_data.append(data_temp)
#
# predict_data = np.array(predict_data)
# print(predict_data,predict_data.shape)
# # # np.save('predict_data.npy',predict_data)

# (array([21, 23, 25, 33, 37, 41], dtype=int64),)0
# (array([ 7,  9, 11, 13, 15, 17, 27, 29, 31], dtype=int64),)1
# (array([42, 44], dtype=int64),)2
# (array([18, 38, 46], dtype=int64),)3
# (array([ 5, 19, 43, 45, 47], dtype=int64),)4
# (array([ 6,  8, 10, 12, 14, 16], dtype=int64),)5
# (array([20, 22, 24, 32, 36, 40], dtype=int64),)6
# (array([28, 30, 34], dtype=int64),)7
# (array([ 0,  2,  4, 26], dtype=int64),)8
# (array([ 1,  3, 35, 39], dtype=int64),)9

x1,y1,z1=[],[],[]

for index, n_neighbors in enumerate((11,12,13,14,15,16,17,18,19,20)):  # 4,5,6,7,8,9,10,11,12,13,14,15,16
    for index, k in enumerate((10,11,12,13,14,15,16,17,18,19,20)):
        y_pred = SpectralClustering(affinity='nearest_neighbors',n_clusters=k, n_neighbors=n_neighbors).fit_predict(X)
        print ("Calinski-Harabasz Score with n_neighbors=", n_neighbors, "n_clusters=", k,"score:", metrics.calinski_harabaz_score(X, y_pred) )
        x1.append(n_neighbors)
        y1.append(k)
        z1.append(metrics.calinski_harabaz_score(X, y_pred))

print(x1,y1,z1)

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

# 生成画布、3D图形对象、三维散点图
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x1,y1,z1)

# 设置坐标轴显示以及旋转角度
ax.set_xlabel('谱聚类中心数量')
ax.set_ylabel('KNN近邻k值')
ax.set_zlabel('CH指标得分')
ax.view_init(elev=10,azim=255)
plt.show()
