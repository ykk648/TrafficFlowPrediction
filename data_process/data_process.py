#!/usr/bin/env python
# -*- coding: utf-8 -*-

from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import calendar
import os
import func

# 处理一个截面  两车道
def main_data_process(node_lists):
    # node_lists = [[961,962,1462],[959,960,1459]]

    section_data = []

    for node_list in node_lists:
        lane_data = np.zeros(shape=(2016,))
        for node_name in node_list:
            sensor_node = '/'+str(node_name)+'.v30'
            sensor_node_data = func.get_data(sensor_node)
            print(sensor_node_data)
            lane_data += sensor_node_data

        # func.plot_7_day(lane_data)
        section_data.append(lane_data.tolist())

    section_data = np.array(section_data)
    print(section_data,section_data.shape)

    # np.savetxt('a.csv',section_data,delimiter=',')
    return section_data


all_node_lists = [[[961,962,1462],[959,960,1459]],   # I94
                  [[978, 979, 980], [976, 977, 3149]],
                  [[987, 988, 989,990], [983, 984, 985,986]],
                  [[481,482,483,484], [476,477,478,479]],
                  [[472,473,474,475], [468,469,470,471]],
                  [[451,452,453,454], [507,508,509,510]],
                  [[90,91,92,93], [86,87,88,89]],
                  [[65,66,67,68,69], [59,60,61,62,63]],
                  [[54,55,56,57,58], [49,50,51,52,53]],
                  [[365,366,367,368], [413,414,415,419]],
                  [[3062,3063],[2969,2970] ],   # I169
                  [[3056,3057],[2975,2976]],   #  2975 2976 离群点剔除
                  #[ [3041,3042]], #  24离群点被踢掉 [2991,2992]
                  [[3028,3029,3068],[3004,3005]],
                  [[2053,2054,4203],[2064,2065,4252]],  # TH100
                  [[4210,4211,4212],[4243,4244,4245]],
                  [[4213,4214,4215],[4240,4241,4242]],
                  [[4222,4223,4224],[4231,4232,4233]],
                  [[1771,1772],[1693,1694,1695]],# I394   没有1773 1696
                  [[1765,1766],[1698,1699]],  # 1767 1700
                  [[1760,1761,1762],[1702,1703,1704]], # 1763 1705
                  [[1754,1755,1756],[1708,1709,1710]],
                  [[1740,1741,1742,1733],[1734,1730,1731,1732]],
                  [[788,789,790,791],[792,793,794,795]],
                  [[777,778,779,780],[781,783,784,785]],
                  # [[767,768,769,773],[1737,770,771,772,774]]
                  ]

print(len(all_node_lists))


count=0
for i in range(len(all_node_lists)):
    for j in range(len(all_node_lists[0])):
        for k in range(len(all_node_lists[0][0])):
            count+=1
print(count)

# 考虑到双车道
# all_section_data = []
# for node_list in all_node_lists:
#     section_data = main_data_process(node_list)
#     # 合成 存储
#     # print(section_data)
#     all_section_data.append(section_data.tolist())
# all_section_data = np.array(all_section_data)
# print(all_section_data)
# print(all_section_data.shape)

# 单车道
all_section_data = []
for node_lists in all_node_lists:
    for node_list in node_lists:
        lane_data = np.zeros(shape=(6048,))
        for node_name in node_list:
            sensor_node = '/'+str(node_name)+'.v30'
            sensor_node_data = func.get_data(sensor_node)
            print(sensor_node_data)
            lane_data += sensor_node_data
        # section_data = main_data_process(node_list)
        # 合成 存储
        # print(section_data)
        all_section_data.append(lane_data.tolist())
all_section_data = np.array(all_section_data)
print(all_section_data)
print(all_section_data.shape)

# all_section_data.tofile('data_single_lane.csv',sep=',',format='%d')
# np.save('data_single_lane.npy',all_section_data)