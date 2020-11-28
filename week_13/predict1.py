import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import random
import jieba
import re

from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import BayesianRidge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor

# 数据加载
content = pd.read_excel('./jobs_4k.xls')
# print(content.columns)
# ['id', 'positionName', 'district', 'stationname', 'jobNature',
#        'companyLabelList', 'industryField', 'salary', 'companySize',
#        'skillLables', 'createTime', 'companyFullName', 'workYear', 'education',
#        'positionAdvantage', 'url', 'detail', 'type']
position_names = content['positionName'].tolist()
skill_labels = content['skillLables'].tolist()

skill_position_graph = defaultdict(list)
for p, s in zip(position_names, skill_labels):
    skill_position_graph[p] += eval(s)

# 设置中文字体为黑体字

plt.rcParams['font.sans-serif'] = ['SimHei']
# 用来正常显示负号
plt.rcParams['axes.unicode_minus'] = False

# 以30个随机选择的工作职位作为可视化

sample_nodes = random.sample(position_names, k=30)
sample_nodes_connections = sample_nodes
for p, skills in skill_position_graph.items():
    if p in sample_nodes:
        sample_nodes_connections += skills

#
G = nx.Graph(skill_position_graph)
sample_graph = G.subgraph(sample_nodes_connections)
plt.figure(figsize=(30, 20))
pos = nx.spring_layout(sample_graph, k=1)  # K=1，nodes间的距离
nx.draw(sample_graph, pos, with_labels=True, node_size=30, font_size=10)
# plt.show()

# 使用pagerank算法，对核心能力和职位进行排序
pr = nx.pagerank(G, alpha=0.85)
ranked_position_and_ability = sorted([(name, value) for name, value in pr.items()], key=lambda x: x[1], reverse=True)
# print(ranked_position_and_ability)

# 特征X，去掉salary
X_content = content.drop(['salary'], axis=1)

target = content['salary'].tolist()
X_content['merged'] = X_content.apply(lambda x: ''.join(str(x)), axis=1)
# print(X_content['merged'])

# 转换为list
X_string = X_content['merged'].tolist()


# print(X_string)
#
def get_one_row_job_string(x_string_row):
    job_string = ''
    for i, element in enumerate(x_string_row.split('\n')):
        if len(element.split()) == 2:
            _, value = element.split()
            #
            if i == 0:
                continue
            job_string += value
    return job_string


cutted_X = []


def token(string):
    return re.findall('\w+', string)


for i, row in enumerate(X_string):
    job_string = get_one_row_job_string(row)
    cutted_X.append(' '.join(list(jieba.cut(''.join(token(job_string))))))

# print(cutted_X)



vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(cutted_X)

# 平均值
target_numical = [np.mean(list(map(float, re.findall('\d+', s)))) for s in target]
# print(type(target_numical))
Y = target_numical


# 使用贝叶斯模型
model_bayes = BayesianRidge(compute_score=True)
model_bayes.fit(X.toarray(), Y)

# 使用KNN模型
model_knn = KNeighborsRegressor(n_neighbors=2)
model_knn.fit(X, Y)
# print(model.score)


model_lr = LinearRegression()
model_lr.fit(X, Y)

model_svr = SVR(kernel='rbf')
model_svr.fit(X, Y)

#
model_rf = RandomForestRegressor(max_depth=1, random_state=0)
model_rf.fit(X, Y)

model_gbrt = GradientBoostingRegressor(random_state=1)
model_gbrt.fit(X, Y)

model_dt = DecisionTreeRegressor(max_depth=4)
model_dt.fit(X, Y)

model_et = ExtraTreeRegressor(random_state=0)
model_et.fit(X, Y)

def predict_by_label(test, model):
    test_words = list(jieba.cut(test))
    test_vec = vectorizer.transform(test_words)
    predict_value = model.predict(test_vec)
    return predict_value[0]


test = '测试 北京 3年 专科'
print(test, predict_by_label(test, model_gbrt))
test2 = '测试 北京 4年 专科'
print(test2, predict_by_label(test2, model_gbrt))
test3 = '算法 北京 4年 本科'
print(test3, predict_by_label(test3, model_gbrt))
test4 = 'UI 北京 4年 本科'
print(test4, predict_by_label(test4, model_gbrt))
persons = ["广州Java本科3年掌握大数据",
           "沈阳Java硕士3年掌握大数据",
           "沈阳Java本科3年掌握大数据",
           "北京算法硕士3年掌握图像识别"]
for p in persons:
    print('{}的薪水AI预测的结果是{}'.format(p, predict_by_label(p, model_gbrt)))


test_words = list(jieba.cut(test))
test_vec = vectorizer.transform(test_words)
model_bayes_predict = model_bayes.predict(test_vec)
model_knn_predict = model_knn.predict(test_vec)
model_lr_predict = model_lr.predict(test_vec)
model_rf_predict = model_rf.predict(test_vec)
model_gbrt_predict = model_gbrt.predict(test_vec)
model_dt_predict = model_dt.predict(test_vec)
model_et_predict = model_et.predict(test_vec)

plt.figure()
plt.plot(model_bayes_predict, 'gd', label='BayesianRidge')
plt.plot(model_knn_predict, 'b^', label='KNeighborsRegressor')
plt.plot(model_lr_predict, 'ys', label='LinearRegression')
plt.plot(model_rf_predict, 'r*', ms=10, label='RandomForestRegressor')
plt.plot(model_gbrt_predict, 'g^', ms=10, label='GradientBoostingRegressor')
plt.plot(model_dt_predict, 'rs', ms=10, label='DecisionTreeRegressor')
plt.plot(model_et_predict, 'go-', ms=10, label='ExtraTreeRegressor')

plt.tick_params(axis='x', which='both', bottom=False, top=False,
                labelbottom=False)
plt.ylabel('predicted')
plt.xlabel('training samples')
plt.legend(loc="best")
plt.title('Regressor predictions and their average')

plt.show()


# SVR回归
# 测试 北京 3年 专科 14.280236627299642
# 测试 北京 4年 专科 14.280236627299642
# 算法 北京 4年 本科 21.207578840859565
# UI 北京 4年 本科 18.25639345119578
# 广州Java本科3年掌握大数据的薪水AI预测的结果是20.907599057406905
# 沈阳Java硕士3年掌握大数据的薪水AI预测的结果是20.85874204893977
# 沈阳Java本科3年掌握大数据的薪水AI预测的结果是20.85874204893977
# 北京算法硕士3年掌握图像识别的薪水AI预测的结果是23.512820809788508

# LinearRegression
# 测试 北京 3年 专科 14.24395211636426
# 测试 北京 4年 专科 14.24395211636426
# 算法 北京 4年 本科 20.945851067381216
# UI 北京 4年 本科 15.55542280986383
# 广州Java本科3年掌握大数据的薪水AI预测的结果是25.844682434012405
# 沈阳Java硕士3年掌握大数据的薪水AI预测的结果是21.51814408973893
# 沈阳Java本科3年掌握大数据的薪水AI预测的结果是21.51814408973893
# 北京算法硕士3年掌握图像识别的薪水AI预测的结果是32.09468163978047

# RandomForestRegressor
# 测试 北京 3年 专科 20.093091478797827
# 测试 北京 4年 专科 20.093091478797827
# 算法 北京 4年 本科 20.093091478797827
# UI 北京 4年 本科 20.093091478797827
# 广州Java本科3年掌握大数据的薪水AI预测的结果是20.093091478797827
# 沈阳Java硕士3年掌握大数据的薪水AI预测的结果是20.093091478797827
# 沈阳Java本科3年掌握大数据的薪水AI预测的结果是20.093091478797827
# 北京算法硕士3年掌握图像识别的薪水AI预测的结果是20.093091478797827

# GradientBoostingRegressor
# 测试 北京 3年 专科 14.117535155736773
# 测试 北京 4年 专科 14.117535155736773
# 算法 北京 4年 本科 15.96908973303177
# UI 北京 4年 本科 14.987420876896879
# 广州Java本科3年掌握大数据的薪水AI预测的结果是15.96908973303177
# 沈阳Java硕士3年掌握大数据的薪水AI预测的结果是15.96908973303177
# 沈阳Java本科3年掌握大数据的薪水AI预测的结果是15.96908973303177
# 北京算法硕士3年掌握图像识别的薪水AI预测的结果是16.94915248003367

# DecisionTreeRegressor
# 测试 北京 3年 专科 15.164248704663212
# 测试 北京 4年 专科 15.164248704663212
# 算法 北京 4年 本科 15.164248704663212
# UI 北京 4年 本科 15.164248704663212
# 广州Java本科3年掌握大数据的薪水AI预测的结果是15.164248704663212
# 沈阳Java硕士3年掌握大数据的薪水AI预测的结果是15.164248704663212
# 沈阳Java本科3年掌握大数据的薪水AI预测的结果是15.164248704663212
# 北京算法硕士3年掌握图像识别的薪水AI预测的结果是15.164248704663212

# ExtraTreeRegressor
# 测试 北京 3年 专科 12.5
# 测试 北京 4年 专科 12.5
# 算法 北京 4年 本科 12.5
# UI 北京 4年 本科 12.5
# 广州Java本科3年掌握大数据的薪水AI预测的结果是17.5
# 沈阳Java硕士3年掌握大数据的薪水AI预测的结果是12.5
# 沈阳Java本科3年掌握大数据的薪水AI预测的结果是12.5
# 北京算法硕士3年掌握图像识别的薪水AI预测的结果是12.5
