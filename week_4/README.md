## Thinking1: ALS都有哪些应用场景
ALS(Alternating least squares)可以应用在推荐系统上。  
- 利用ALS可以根据用户历史打分信息，得到预测用户对新电影、电视剧的评分，从而对用户进行影视作品的推荐。
- 利用ALS可以得到用户在购物时对物品的行为，预测出用户可能会喜爱的物品，从而对用户进行商品的推荐。

## Thinking2: ALS进行矩阵分解的时候，为什么可以并行化处理
优化的目标函数可以写为：  
$$J(x,y) = min \sum (r_{ui} - x_u^Ty_i)^2 + \lambda(\sum \Vert x_u \Vert_2^2 + \sum \Vert y_i \Vert_2^2)$$  
ALS的第一步是将y固定住，然后求解x的梯度来更新x，这里函数为：  
$$\frac {\partial J(x_u)} {\partial x_u} = -2Y_u(R_u - Y_u^Tx_u) + 2\lambda x_u^2 = 0$$  
求解得到：  
$$x_u = (Y_u Y_u^T + \lambda I)^{-1} Y_u R_u$$  
下一步同理，固定x，求y的梯度来更新y:  
$$y_i = (X_i X_i^T + \lambda I)^{-1} X_i R_i$$  
可以看到，在更新$x_u,y_i$的过程中，每个时刻只依赖自身，不依赖于其他标的物的特征向量，所以可以将不同的$x_i,y_i$放在不同的服务器上执行。

## Thinking3: 梯度下降法中的批量梯度下降（BGD），随机梯度下降（SGD），和小批量梯度下降有什么区别（MBGD）

### 批量梯度下降（Batch Gradient Descent，简称BGD）
- 批量梯度下降法是最原始的形式，它是指在每一次迭代时使用所有样本来进行梯度的更新。
- 当目标函数为凸函数时，BGD一定能够得到全局最优
- 速度慢

### 随机梯度下降（Stochastic Gradient Descent，简称SGD）
- 随机梯度下降是每次迭代使用一个样本来对参数进行更新，用它来代替整体
- 最终收敛值在最优解附近
- 速度快

### 小批量梯度下降（Mini-Batch Gradient Descen，简称MBGD）
- 小批量梯度下降，是对批量梯度下降以及随机梯度下降的一个折中办法。其思想是：每次迭代使用 ** batch_size** 个样本来对参数进行更新
- 收敛速度适中，正确率介于BGD, SGD中间
- 可实现并行化

### Thinking4 你阅读过和推荐系统/计算广告/预测相关的论文么？有哪些论文是你比较推荐的，可以分享到微信群中
推荐系统的论文我读的比较少，但是我读过一本书项量编著的《推荐系统实践》，推荐给大家。

WINDOWS 10环境下的Pyspark配置 （基于Anaconda环境）：https://zhuanlan.zhihu.com/p/37617055