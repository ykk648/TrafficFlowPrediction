#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import matplotlib


plt.figure()

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 30,
}

plt.subplot(1,3,1)

x=['LSTM','SAEs','LSTM-SAEs']
y=[0.9634,0.9605,0.9767]
plt.bar(x,y,width=0.8,color=['teal','c','coral'])
plt.ylim([0.95,1])
plt.xticks(rotation=20)
#plt.ylabel(u"融合参数",fontproperties = myfont)
#plt.xlabel(u"评价指标",fontproperties = myfont)
plt.ylabel("Explained Variance Score")
# plt.xlabel("Model")


plt.subplot(1,3,2)

x=['LSTM','SAEs','LSTM-SAEs']
y=[13.87,15.28,12.54]
plt.bar(x,y,width=0.8,color=['teal','c','coral'])
plt.ylim([10,20])
plt.xticks(rotation=20)
#plt.ylabel(u"融合参数",fontproperties = myfont)
#plt.xlabel(u"评价指标",fontproperties = myfont)
plt.ylabel("MAPE")
plt.xlabel("Model")




plt.subplot(1, 3, 3)
x=['LSTM','SAEs','LSTM-SAEs']
y=[55.79,60.69,53.37]
plt.bar(x,y,width=0.8,color=['teal','c','coral'])
plt.ylim([50,65])
plt.xticks(rotation=20)
#plt.ylabel(u"融合参数",fontproperties = myfont)
#plt.xlabel(u"评价指标",fontproperties = myfont)
plt.ylabel("MSE")
# plt.xlabel("Model")



plt.tight_layout()

plt.show()
