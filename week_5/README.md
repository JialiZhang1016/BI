## Thinking1 在实际工作中，FM和MF哪个应用的更多，为什么？
**MF: Matrix Factorization 矩阵分解**  
目标函数：
$$J(x,y) = min \sum (r_{ui} - x_u^Ty_i)^2 + \lambda(\sum \Vert x_u \Vert_2^2 + \sum \Vert y_i \Vert_2^2)$$

**FM: Factorization Machines 因子分解机**  
目标函数：
$$\hat y(x) = w_0 + \sum w_n x_n$$  
$$\hat y(x) = w_0 + \sum w_n x_n + \sum \sum w_{ij} x_i x_j$$  
$$\hat w_{ij} = \langle V_i,V_j \rangle$$  
本质上，MF模型是FM模型的特例，MF可以被认为是只有User ID 和Item ID这两个特征Fields的FM模型，MF将这两类特征通过矩阵分解，来达到将这两类特征embedding化表达的目的。而FM继承了MF的特征embedding化表达这个优点，同时引入了更多Side information作为特征，将更多特征及Side information embedding化融入FM模型中。所以很明显FM模型更灵活，能适应更多场合的应用范围。

## Thinking2 FFM与FM有哪些区别？
**FFM: Field Factorization Machines**
1. FFM是在FM的基础上加上了`field`的概念，把相同性质的特征归于同一field，比如时间：`2020/5`，`2020/6`属于同一field。FM可以看作是只有一个场的FFM
2. FM算法每个特征只有一个隐向量，FFM算法每个特征有多个隐向量
3. FFM的算法复杂度为$O(kn^2)$；FM的算法复杂度为$O(kn)$

## Thinking3 DeepFM相比于FM解决了哪些问题，原理是怎样的
**DeepFM: Deep Neural Networks + Factorization Machines**

**原理：**DeepFM包括FM和DNN两个部分，先对Sparse Features的稀疏矩阵做Dense Embedding，结果由FM和DNN共享，FM负责提取低维特征，DNN负责提取高维特征。最终的预测由两部分输出的相加得到：

$$\hat y = sigmoid(y_{FM} + y_{DNN})$$

**解决的问题：**FM可以做特征组合，但是计算量大，一般只考虑两阶特征组合。而DeepFM可以很好的解决这个问题：

## Thinking4 Surprise工具中的baseline算法原理是怎样的？ BaselineOnly和KNNBaseline有什么区别？
**baseline原理**：  
surprise中的baseline算法的目标函数是：

$$\sum(r_{ui} - (\mu + b_u + b_i))^2 + \lambda (b_u^2 + b_i^2)$$

通过最小化目标函数得到$b_u$, $b_i$。其中最小化目标函数的方法有ALS和SGD两种。

**BaselineOnly和KNNBaseline的区别：**  
BaselineOnly是surprise中实现baseline算法的方法，算法原理如上。而KNNBaseline的算法原理是一种考虑了基线评分的基于邻域的算法。其中
$$\hat r_{ui} = b_{ui} + \frac {\sum sim(u,v)*(r_{vi} - b_{vi})} {\sum sim(u,v)}$$
是基于用户邻域的推荐算法
$$\hat r_{ui} = b_{ui} + \frac {\sum sim(i,j)*(r_{uj} - b_{uj})} {\sum sim(i,j)}$$
是基于物品邻域的推荐算法。具体在代码中通过以下来实现。
```{python}
sim_options = {'name': 'cosine',
               'user_based': False  # compute  similarities between items
               }
algo = KNNBasic(sim_options=sim_options)
```

## Thinking5 基于邻域的协同过滤都有哪些算法，请简述原理
**UserCF:** 给用户推荐和他兴趣相似的其他用户喜欢的物品  
**ItemCF:** 给用户推荐和他之前喜欢的物品相似的物品，比如slope one算法，







**Reference：**
- https://zhuanlan.zhihu.com/p/58160982
- https://cloud.tencent.com/developer/article/1648749
- https://tech.meituan.com/2016/03/03/deep-understanding-of-ffm-principles-and-practices.html 
- http://fancyerii.github.io/2019/12/19/deepfm/#deepfm
- https://surprise.readthedocs.io/en/stable/prediction_algorithms.html#baseline-estimates-configuration

