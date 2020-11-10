## Thinking1 什么是Graph Embedding，都有哪些算法模型
**DeepWalk**：这是一种基于word2vec的方法，利用randomwalk来将节点间的关系转成类似于语言中句子的一维形式（randomwalk是从node1开始走，随机去邻居节点，然后走到没有邻居节点后停止或到达指定节点数后停止），也就是node1 node2 node3 node4 node5（对应的就是NLP中的W1,W2,W3,W4,W5），对于每个节点可以生成若干个这样的一维关系，即节点序列，且每个节点序列的长度也是可以指定的。之后把这些生成的“句子”看做NLP中的句子，节点看做word，然后进行word2vec操作，从而实现graph embedding。

**node2vec**：是在Deepwalk基础上设定了两个超参数，实际上是定义了跳转外部节点和返回上一节点的概率，通过p和q这两个超参数（p越小，返回上一节点概率越大，q越小，跳转外部节点概率越大）实现节点跳转规则的控制，实现近似于BFS和DFS的节点遍历方式（即让walk部分着重于节点的临节点微观特征或宏观特征）。实际上，当node2vec如果将p和q设置为1，就是deepwalk，因为设置为1时候，去往所有临节点的概率都一样，即随机，也就是randomwalk。

## Thinking2 如何使用Graph Embedding在推荐系统，比如NetFlix 电影推荐，请说明简要的思路
可以将item，user，type这些内容当做graph中的node进行embedding，在推荐的时候，对用户推荐的时候，先找到与该用户相似的用户，再从相似的用户看过的电影中找出待推荐用户未看过的电影。同时也可以用相同的方法，进行itemCF将对应的电影推荐给所需推荐的用户。当然，也可以利用type来找出相似类型的电影推荐给用户。

## Thinking3 数据探索EDA都有哪些常用的方法和工具
常用的方法如下：
- 查看数据前5行：dataframe.head()
- 查看数据的信息，包括每个字段的名称、非空数量、字段的数据类型：data.info()
- 查看数据的统计概要（count/mean/std/min/25%/50%/75%max）：data.describe()
- 查看dataframe的大小：dataframe.shape
- 查看数据按某个特征的分组统计
- 对数据进行可视化，包括数据分布，缺失值可视化

常用的工具如下： 
- msno: 缺失值可视化
- pandas_profiling: 一行代码生成报告