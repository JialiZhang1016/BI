## 1. Introduction
**Aim of this paper: 证明 slope one 可以满足以下5个目标。**
1. 易维护
2. 易更新
3. 查询时间短
4. 对之前用户的依赖小
5. 准确，可解释

**Contirbution:**
提出一种新方法slope one, 这种方法更适合CF task

## 2. Related work
**Memory-Based 缺点：**
- 可伸缩性，对稀疏数据的敏感性差
- 查询速度慢
- 必须依赖之前的用户行为

**Model-Based 缺点：**
查询速度快，但是吃内存

## 3. 本文提出了三种CF schemes:
**会与四种参考方案进行对比**
1. per user average
2. bias form mean
3. adjusted cosine item based (model based scheme)
4. **pearson scheme (memory based scheme)**

#### 3.1 The slope one scheme
Our Slope One algorithms work on the intuitive principle of a **`popularity differential`** between items for users. 

#### 3.2 The weighted slope one scheme
在slope one 的基础上，使用评分的个数进行加权

#### 3.3 The bi-polar slope one scheme
在weighted slope one scheme 的基础上将预测分为两类，一类从“用户喜爱”中抽取，一类从“用户不喜爱”中抽取。"喜爱"与"不喜爱"的分界点是5，when scale is 0-10.

**其实这种方法更加合理，因为用户评分的密度函数并不是均匀或者呈现整体分布的。通过对豆瓣电影评分的观察，有很大一部分的高分电影都呈现`F`分布。也就是打高分的较多**

**限制：**
1. item aspect: 仅考虑组间的离差（“喜欢”组 / “不喜欢”组）
2. user aspect: 仅考虑同时给两个项目评分，并且两个项目分到同样组的用户

## 4. 实验结果

**数据集**
1. EachMovie data set
rating: 0，0.2，0.4，0.6，0.8，1
2. Movielens data set
rating: 1，2，3，4，5

**训练集和测试集**
50k as traing set
100k as testing set

**结果比较**
1. BIAS FROM MEAN > (ACIB, PUA)
2. slope one > BIAS FROM MEAN
3. BI-polar > weighted > slope one
4. PEARSON = slope one

## 5. 结论
1. slope one 可以实现上文提到的5个好处；
2. **innovation:** splitting ratings into dislike and like subsets can be an effective technique for improving accuracy；
3. 2004.10 Bell/MSN 已经在用weighted slope one scheme了。