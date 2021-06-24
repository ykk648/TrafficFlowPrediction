#!/usr/bin/env python
# -*- coding: utf-8 -*-
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import calendar
import os

sensor_node = '/923.v30'
DIR_PATH = 'E:/PycharmProjects/TDRL/data/'

pathNames = ['E:/PycharmProjects/TDRL/data/20181022/2975.v30',
             'E:/PycharmProjects/TDRL/data/20181022/2976.v30',
             'E:/PycharmProjects/TDRL/data/20181022/3041.v30',
             'E:/PycharmProjects/TDRL/data/20181022/3042.v30']
             # 'E:/PycharmProjects/TDRL/data/20181022/214.v30'

# sensor_node_data = np.empty([1,288])




# r_int_list_sum = []

for pathName in pathNames:
    f = open(pathName,'rb')
    records = iter(partial(f.read, 1), b'')
    i=0
    r_int_list = []
    r_sum = 0
    # 每5分钟
    for r in records:
        r_int = int.from_bytes(r, byteorder='big', signed=True)
        i+=1
        if 0<=r_int<=40:
            r_sum+=r_int
        else:
            print('ont bad'+str(r_int))
        if i==10:
            r_int_list.append(r_sum)
            i=0
            r_sum=0
        # print(r)
        # print(r_int)
    print(r_int_list)
    print(len(r_int_list))

    y = np.array(r_int_list)
    x = np.array(range(288))


    time_list = ['00:00', '02:00', '04:00', '06:00', '08:00',
                 '12:00','14:00','16:00','18:00','20:00','22:00','24:00']

    # plt.xticks( range(12), calendar.month_name[1:13], rotation=17 )
    aa = np.linspace(0, 288, 12)
    plt.xticks( aa, time_list, rotation=0)
    # plt.plot(x,y,'r')
    plt.bar(x,y,1,alpha=1,color='b')
    plt.show()