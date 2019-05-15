# 项目背景
在今天产品高度同质化的阶段，市场竞争不断加剧，企业与企业之间的竞争，主要集中在对客户的争夺上。“用户就是上帝”促使众多企业不惜代价去争夺尽可能多的新客户。但是，在企业不惜代价发展新用户的过程中，往往会忽视老用户的流失情况，结果就导致出现新用户在源源不断的增加，辛苦找来的老用户却在悄无声息的流失的窘状。
如何处理客户流失的问题，成为一个非常重要的课题。那么，我们如何从数据汇总挖掘出有价值的信息，来防止客户流失呢？
行业情况：携程曾占据在线酒店的主要市场，但是随着美团在2017年加入，格局很快被打破
![2018年3月在线酒店预定平台情况](https://upload-images.jianshu.io/upload_images/12564647-8b17cfbc23efc896.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1000/format/webp)

从下图可以看出：行业Top3的用户重合率较低，用户差异度明显。用户与市场竞争具有很大的相关性。

![2018年2月－3月重合用户](https://upload-images.jianshu.io/upload_images/12564647-754111beffa7e2be.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1000/format/webp)

# 项目目标
+ 挖掘用户流失的关键因素
+ 预测客户的转化效果
+ 用户画像
# 项目方案
+ 使用pandas,numpy,sklearn.processing对数据进行预处理
+ 使用LogisticRedression和DecisionTreeClassifier预测
+ 用K-Means对用户进行画像，针对不同用户类别，提出可行建议
# 项目流程
1. 数据预处理
  + 衍生变量
  + 删除缺失比列>80%的列
  + 过滤无用维度
  + 异常值负数处理
  + 缺失值填充
  + 极值处理
  + 相关性
  + 标准化处理
2. 建模预测
  + LogisticRedression预测
  + DecisionTreeClassifier预测
  + 结果可视化
3. 用户画像
  + 用K-Means对用户进行画像
