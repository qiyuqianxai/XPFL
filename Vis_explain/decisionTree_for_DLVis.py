# 标准库
import os
import numpy as np
import pandas as pd
from torchvision import datasets
# 可视化的库
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
# 建模和机器学习
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# 读取数据
dataset_path = r"D:\datasets"
dataset_name = "cifar10"
train_data = datasets.CIFAR10(root=os.path.join(dataset_path,dataset_name), download=True, train=True)
test_data = datasets.CIFAR10(root=os.path.join(dataset_path,dataset_name),  download=True, train=False)
print(train_data.classes,type(train_data.classes))
X_train = np.array(train_data.data).reshape(60000,-1)
Y_train = np.array(train_data.targets).reshape(60000,-1)
X_val = np.array(test_data.data).reshape(10000,-1)
Y_val = np.array(test_data.targets).reshape(10000,-1)
print(X_train.shape,Y_train.shape)
print(train_data.classes,type(train_data.classes))
# 将训练集与验证集的尺度进行输出
print('Shape of X_train:', X_train.shape)
print('Shape of y_train:', Y_train.shape)
print('Shape of X_valid:', X_val.shape)
print('Shape of y_valid:', Y_val.shape)
dtModel = DecisionTreeClassifier()  # 建立模型
dtModel.fit(X_train,Y_train)
# 测试决策树
prediction = dtModel.predict(X_val)
plt.figure(figsize=(10,7))
cm = confusion_matrix(Y_val,  prediction)

ax = sns.heatmap(cm, annot=True, fmt="d",cmap='Blues')
plt.ylabel('Actual label')  # x轴标题
plt.xlabel('Predicted label')  # y轴标题

acc = accuracy_score(Y_val,prediction)
print(f"Sum Axis-1 as Classification accuracy: {acc* 100}")

# 决策树可解释化
from sklearn import tree
import pydotplus

dot_data = tree.export_graphviz(dtModel, out_file="VisData.dot",
                                class_names=train_data.classes,
                                filled=True, rounded=True,
                                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data)
graph.write_pdf('VisData.pdf')
