"""
Processing the data
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def process_data(train, test, lags):
    """Process data
    Reshape and split train\test data.

    # Arguments
        train: String, name of .csv train file.
        test: String, name of .csv test file.
        lags: integer, time lag.
    # Returns
        X_train: ndarray.
        y_train: ndarray.
        X_test: ndarray.
        y_test: ndarray.
        scaler: StandardScaler.
    """
    attr = 'Lane 1 Flow (Veh/5 Minutes)'
    df1 = pd.read_csv(train, encoding='utf-8').fillna(0)
    df2 = pd.read_csv(test, encoding='utf-8').fillna(0)

    # scaler = StandardScaler().fit(df1[attr].values)
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1[attr].values.reshape(-1, 1))
    flow1 = scaler.transform(df1[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    flow2 = scaler.transform(df2[attr].values.reshape(-1, 1)).reshape(1, -1)[0]

    train, test = [], []
    for i in range(lags, len(flow1)):
        train.append(flow1[i - lags: i + 1])
    for i in range(lags, len(flow2)):
        test.append(flow2[i - lags: i + 1])

    train = np.array(train)
    test = np.array(test)
    np.random.shuffle(train)

    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]

    return X_train, y_train, X_test, y_test, scaler


def process_data_combine(train, test, lags, num):

    # attr = 'Lane 1 Flow (Veh/5 Minutes)'
    # df1 = pd.read_csv(train, encoding='utf-8').fillna(0)
    # df2 = pd.read_csv(test, encoding='utf-8').fillna(0)
    #
    # # scaler = StandardScaler().fit(df1[attr].values)
    # scaler = MinMaxScaler(feature_range=(0, 1)).fit(df1[attr].values.reshape(-1, 1))
    # flow1 = scaler.transform(df1[attr].values.reshape(-1, 1)).reshape(1, -1)[0]
    # flow2 = scaler.transform(df2[attr].values.reshape(-1, 1)).reshape(1, -1)[0]

    train = []

    X = np.load('E:/PycharmProjects/TDRL/predict_data.npy')
    Y = np.load('E:/PycharmProjects/TDRL/data_single_lane.npy')

    # flow1_data = X[:,:1152]

    a = X[:, :1440]
    b = X[:, 2016:3456]
    c = X[:, 4032:5472]

    flow1_data = np.hstack((a, b, c))

    # flow2_data = X[:, 1152:1440]

    # flow2_data = Y[51, 1140:1440]  # 聚类前的值 -300

    length = len(flow1_data)

    # scaler = StandardScaler().fit(X[0].reshape(-1, 1))
    # scaler = MinMaxScaler(feature_range=(0, 1)).fit(X[0].reshape(-1, 1))
    scaler = MinMaxScaler(feature_range=(0, 1)).fit(X.reshape(-1, 1))

    for i in range(length):
        # tt = len(flow1_data[i])

        # scaler = MinMaxScaler(feature_range=(0, 1)).fit(flow1_data[i].reshape(-1, 1))

        flow1_data[i] = scaler.transform(flow1_data[i].reshape(-1, 1)).reshape(1, -1)[0]

        # flow2_data[i] = scaler.transform(flow2_data[i].reshape(-1, 1)).reshape(1, -1)[0]

        for j in range(lags, len(flow1_data[i])):
            train.append(flow1_data[i][j - lags: j + 1])
        # for j in range(lags,len(flow2_data[i])):
        #     test.append(flow2_data[i][j - lags: j + 1])
    train = np.array(train)

    # # 求平均效果使用的test集
    # X_test,y_test = [],[]
    # for j in range(len(Y)):
    #     test = []
    #
    #     flow2_data = Y[j, (5184-lags):5472] # 周五
    #
    #     # flow2_data = Y[j, 5460:5760]  # 周六
    #     # flow2_data = Y[j, 5748:6048] # 周天
    #
    #     flow2_data = scaler.transform(flow2_data.reshape(-1, 1)).reshape(1, -1)[0]
    #     for j in range(lags, len(flow2_data)):
    #         test.append(flow2_data[j - lags: j + 1])
    #     test = np.array(test)
    #     X_test.append(test[:, :-1])
    #     y_test.append(test[:, -1])

    # 原始单一道路使用的test集
    test = []
    flow2_data = Y[num, 5160:5472]
    flow2_data = scaler.transform(flow2_data.reshape(-1, 1)).reshape(1, -1)[0]
    for j in range(lags, len(flow2_data)):
        test.append(flow2_data[j - lags: j + 1])
    test = np.array(test)
    X_test = test[:, :-1]
    y_test = test[:, -1]

    np.random.shuffle(train)

    X_train = train[:, :-1]
    y_train = train[:, -1]

    return X_train, y_train, X_test, y_test, scaler