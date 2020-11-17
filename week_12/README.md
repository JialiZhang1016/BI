## Thinking1 GCN/Graph Embedding 都有哪些应用场景
- GCN是指Graph Convolution Networks,图卷积网络。实际上跟CNN的作用一样，就是一个特征提取器，只不过它的对象是图数据。
- Graph embedding指的是一种将复杂的网络关系映射成固定维度的特征向量的一种降维技术。
- 他们的应用场景有很多，比如比如商品推荐，金融风控（蚂蚁金服案例），文本分类，聊天机器人语义分析及意图识别，交通流量预测，恶意软件检测等。

参考：https://mhy12345.xyz/technology/graph-embedding-tutorials-introduction/

## Thinking2 在交通流量预测中，如何使用Graph Embedding，请说明简要的思路
- node: 路口/交通信号灯所在位置的经纬度数据
- edge: node与node之间的每一条道路
- weight of edge: 采集到的车流量历史记录/道路特征/日期、时间特征等进行融合构成边的权重。另外，这是一个有向图

## Thinking3 在文本分类中，如何使用Graph Embedding，请说明简要的思路
- node: 关键词/文档
- edge: 信息共现，例如词与词共同出现
- weight of edge: 共同出现的次数，文档间的引用关系（如果有的话）