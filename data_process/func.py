#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np
from functools import partial
import matplotlib.pyplot as plt

def get_data(sensor_node):
    DIR_PATH = 'E:/PycharmProjects/TDRL/data/'
    pathNames = []
    # sensor_node_data = np.empty([1,288])

    for dirName in os.listdir(DIR_PATH):
        pathName = os.path.join(DIR_PATH, dirName)
        if os.path.isdir(pathName):
            pathName += sensor_node
            # print(pathName)
            pathNames.append(pathName)
    print(pathNames[:1])
    r_int_list_sum = []

    r_int_list = []

    for pathName in pathNames:
        f = open(pathName, 'rb')
        records = iter(partial(f.read, 1), b'')
        i = 0
        r_sum = 0
        # 每5分钟
        for r in records:
            r_int = int.from_bytes(r, byteorder='big', signed=True)
            i += 1
            if 0<=r_int<=40 :
                r_sum += r_int
            else:
                # print('one error node')
                r_sum += 0
            if i == 10:
                r_int_list.append(r_sum)
                i = 0
                r_sum = 0
            # print(r)
            # print(r_int)

        # print(r_int_list, len(r_int_list))
        # r_int_list_sum.append(r_int_list)

    temp = np.array(r_int_list)  # temp.tolist()
    print(temp, temp.shape)
    return temp
    # print(temp.reshape(1,1152))
    # print(temp.shape)

def plot_7_day(lane_data):
    y = lane_data
    x = np.array(range(2016))

    time_list = ['00:00', '02:00', '04:00', '06:00', '08:00',
                 '12:00', '14:00', '16:00', '18:00', '20:00', '22:00', '24:00']

    # plt.xticks( range(12), calendar.month_name[1:13], rotation=17 )
    aa = np.linspace(0, 2016, 12)
    plt.xticks(aa, time_list, rotation=0)
    # plt.plot(x,y,'r')
    plt.bar(x, y, 1, alpha=1, color='b')
    plt.show()
