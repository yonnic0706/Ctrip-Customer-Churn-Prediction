'''python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn import tree
from sklearn.cluster import KMeans

#导入数据
data = pd.read_csv('userlostprob_train.txt',sep='\t')

'''
查看数据
#查看数据信息
data.info()

#查看前5行数据
data.head()

#查看数值型数据描述统计信息
data.describe()

#查看各列最大值最小值
for i in data.columns:
    print(i,data[i].max(),data[i].min())
    
'''
#查看各列缺失情况，并统计
data_count = data.count()
na_count = len(data) - data_count
na_rate = na_count/len(data)
a = na_rate.sort_values(ascending=True) #按values正序排列，不放倒序是为了后边的图形展示排列
a1 = pd.DataFrame(a)

#绘图查看缺失情况   
#用来正常展示中文标签
plt.rcParams['font.sans-serif']=['SimHei']
x = data.shape[1]
fig = plt.figure(figsize=(8,12)) #图形大小
plt.barh(range(x),a1[0],color='steelblue',alpha=1) 
plt.xlabel('数据缺失占比') #添加轴标签
columns1 = a1.index.values.tolist() #列名称
plt.yticks(range(x),columns1)
plt.xlim([0,1]) #设置X轴的刻度范围
for x,y in enumerate(a1[0]):
    plt.text(y,x,'%.3f' %y,va='bottom')
plt.show()

'''
数据预处理
1.衍生变量
2.删除80%+缺失值列
3.过滤无用维度
4.异常值负数处理
5.缺失值填充
6.极值处理
'''

#1.衍生变量
#添加新列：提前预订 = 入住时间-访问时间
#入住时间和访问时间数据格式为’object，需更换
data['d'] = pd.to_datetime(data['d'])
data['arrival'] = pd.to_datetime(data['arrival'])
data['Advance booking'] = (data['arrival']-data['d']).dt.days

#2.删除缺失值比列88%的列historyvisit_7ordernum
data = data.drop(['historyvisit_7ordernum'],axis=1)

#3.过滤无用的维度
filter_feature = ['sampleid','d','arrival'] #这些维度没有用
feature = []
for x in data.columns:
    if x not in filter_feature:
        feature.append(x)
        
data_x = data[feature]  #共49个特征
data_y = data['label']

#4.异常值负数处的处理
#共6列有负值，其中deltaprice_pre2_t1（24小时内已访问酒店价格与对手价差均值）正常，所以余5列，分别为客户价值（ctrip_profits），客户价值近1年（customer_value_profit）、用户偏好价格-24小时浏览最多酒店价格（delta_price1）、用户偏好价格-24小时浏览酒店平均价格（delta_price2）、当年酒店可订最低价（lowestprice）
#customer_value_profit、ctrip_profits 替换为0
#delta_price1、delta_price2、lowestprice按中位数处理
data_x.loc[data_x.customer_value_profit<0,'customer_value_profit'] = 0
data_x.loc[data_x.ctrip_profits<0,'ctrip_profits'] = 0
data_x.loc[data_x.delta_price1<0,'delta_price1'] = data_x['delta_price1'].median()
data_x.loc[data_x.delta_price2<0,'delta_price2'] = data_x['delta_price2'].median()
data_x.loc[data_x.lowestprice<0,'lowestprice'] = data_x['lowestprice'].median()

#5.缺失值填充
'''
#查看数据分布情况
for i in range(0,48):
    plt.figure(figsize=(2,1),dpi=100)
    plt.hist(data_x[data_x.columns[i]].dropna().get_values())
    plt.xlabel(data_x.columns[i])
plt.show()
'''

#趋于正态分布的字段，用均值填充：字段有businesstate_pre2,cancelrate_pre,businessrate_pre；
fill_list = ['businessrate_pre','businessrate_pre2','cancelrate_pre']
for i in fill_list:
    data_x[i] = data_x[i].fillna(data_x[i].mean())
    
#data_x['businessrate_pre2'] = data_x['businessrate_pre2'].fillna(data_x['businessrate_pre2'].mean())

#右偏分布的字段，用中位数填充
def filling(data):
    for i in range(0,48):
        data_x[data_x.columns[i]] = data_x[data_x.columns[i]].fillna(data_x[data_x.columns[i]].median())
    return data_x

filling(data_x)

'''
#检查缺失值填充情况
data_count2 = data_x.count()
na_count2 = len(data_x) - data_count2
na_rate2 = na_count2/data_count2
aa2 = na_rate2.sort_values(ascending=False)
a3 = pd.DataFrame(aa2)
'''

#6.极值处理
#盖帽法：某连续变量6西格玛之外的记录用正负3西格玛值替代，一般正负3西格玛包含99%的数据，所以默认凡小于百分之一分位数和大于百分之九十九分位数的值用百分之一分位数和百分之九十九分位数代替，俗称盖帽法

data1 = np.array(data_x)  #将data_x替换成numpy格式

for i in range(0,len(data1[0])): #列循环，以便取到每列的值
    a = data1[:,i] #将每列的值赋值到a
    
    b = np.percentile(a,1) #计算1百分位数据
    c = np.percentile(a,99) #计算99百分位数据
    for j in range(0,len(data1[:,0])):
        if a[j] < b:
            a[j] = b
        elif a[j] > c:
            a[j] = c
        else:
            a[j]
            
data1 = pd.DataFrame(data1,columns=feature)  
dat = pd.concat((data1['consuming_capacity'],data1['customer_value_profit'],data1['ordercanncelednum'],data1['ordercanceledprecent'],data1['ctrip_profits'],data1['historyvisit_totalordernum'],data1['lastpvgap'],data1['lasthtlordergap']),axis=1)
dat.to_csv('rfm.csv',index=False)

'''  
#检查处理后数据（极值和负值）
#箱线图检查极值
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
for i in range(0,48):
    plt.figure(figsize=(4,8),dpi=100)
    plt.boxplot(data1[data1.columns[i]].dropna().get_values())
    plt.xlabel(data1.columns[i])
plt.show()
    
#查看最小值是不是负数
list_check = ['ctrip_profits','customer_value_profit','delta_price1','delta_price2','lowestprice']
for i in list_check:
    print(data1[i].min())        
'''    

'''
梳理列与列之间的关系
1.相关性
2.标准化处理
'''

#相关性
#生成相关系数，并保存
corrdf = data1.corr()
corrdf['label'].sort_values(ascending=False)
#去除相关系数小于0.01的特征
delete_columns = []
for i in range(0,corrdf.shape[0]):
    if abs(corrdf.iloc[0,i]) < 0.01:
        delete_columns.append(data1.columns[i])

data1.drop(delete_columns,axis=1,inplace=True)

#去除x与x相关系数大于0.9，且与y的相关呢性值比较小的特征
detele_columns2 = ['historyvisit_totalordernum','delta_price2','cityuvs']
data1.drop(detele_columns2,axis=1,inplace=True)

#标准化处理，是数据更好的适应线性分类模型。
#书上说：数据标准化的目的是：处理不同规模和量纲的数据，使其缩放到相同的数据区间和范围
y = data1['label']
x = data1.drop('label',axis=1)
scaler = preprocessing.StandardScaler()
x = scaler.fit_transform(x)

'''建模'''
#目标变量属于标称型，使用逻辑回归和决策树
#导入模型包，建模


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=.2,random_state=1234)

#逻辑回归模型
lr = LogisticRegression()
lr.fit(x_train,y_train)
y_lr = lr.predict_proba(x_test)[:,1]
fpr_lr,tpr_lr,threshold_lr = metrics.roc_curve(y_test,y_lr)
auc_lr = metrics.auc(fpr_lr,tpr_lr)
score_lr = metrics.accuracy_score(y_test,lr.predict(x_test))
print([score_lr,auc_lr])

#决策树

dtc = tree.DecisionTreeClassifier()
dtc.fit(x_train,y_train)
y_dtc = dtc.predict_proba(x_test)[:,1]
fpr_dtc,tpr_dtc,threshod_dtc = metrics.roc_curve(y_test,y_dtc)
metrics.accuracy_score(y_test,dtc.predict(x_test))

#画图对比结果
plt.rcParams['font.sans-serif']=['SimHei']
fig = plt.plot()
plt.plot(fpr_lr,tpr_lr,label='lr')
plt.plot(fpr_dtc,tpr_dtc,label='dtc')
plt.legend(loc=0) #在合适的地方放置图例
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC carve')
plt.show()


'''K-means用户画像'''
#以下两步放在数据相关性之前，因为相关性内删除粗了一些rfm所需要的列
#dat = pd.concat((data1['consuming_capacity'],data1['customer_value_profit'],data1['ordercanncelednum'],data1['ordercannceledprecent'],data1['ctrip_profits'],data1['historyvisit_7ordernum'],data1['historyvisit_totalordernum'],data1['lastpvgap'],data1['lasthtlordergap']),axis=1)
#dat.to_csv('rfm',index=False)

data = pd.read_csv('rfm.csv')
Kmodel = KMeans(3)
Kmodel.fit(data) #模型训练
data_cluster = Kmodel.cluster_centers_ #查看聚类中心
np.savetxt('cluster_center.csv',data_cluster,delimiter=',')
Kmodel.labels_ #查看样本的类别标签

#统计不同类别样本的数目
r1 = pd.Series(Kmodel.labels_).value_counts()
print('最终每个类别的数目为: \n',r1)
