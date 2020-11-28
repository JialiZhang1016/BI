## Thinking1 KNN与KMeans中的K分别代表什么？
KNN和Kmeans都是很经典的聚类算法.
- KNN的K表示n_neighbors，代表的是邻居的数量，在sklearn中默认值5
- KMeans的K表示n_clusters，代表聚类中心的个数，也就是聚成几类，在sklearn中默认值8

## Thinking2 都有哪些常用的启发式算法？
- 启发式算法：相对于最优化算法提出的，一个问题的最优算法求得该问题每个实例的最优解
- 启发式算法可以这样定义：一个基于直观或经验构造的算法，在可接受的花费（指计算时间和空间）下给出待解决组合优化问题每一个实例的一个可行解，该可行解与最优解的偏离程度一般不能被预计
一般用于解决NP-hard问题，其中NP是指非确定性多项式
- 常用的算法有：模拟退火算法（SA）、遗传算法（GA）、蚁群算法（ACO）、人工神经网络（ANN）

## Thinking3 遗传算法的原理是怎样的？
- 通过模拟自然进化过程(达尔文生物进化论)搜索最优解的方法,遗传操作包括:选择、交叉和变异
- 算法核心:参数编码、初始群体的设定、适应度函数、遗传操作设计、控制参数设定
- 直接对结构对象进行操作,不存在求导和函数连续性的限定
- 采用概率化的寻优方法,不需要确定的规则就能自动获取和指导优化的搜索空间,自适应地调整搜索方向