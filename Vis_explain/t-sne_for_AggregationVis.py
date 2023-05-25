from sklearn import preprocessing
from sklearn.manifold import TSNE
from torchvision import datasets
import os
import numpy as np
import torchvision

# 读取数据
dataset_path = r"D:\datasets"
dataset_name = "MNIST"
train_data = datasets.MNIST(root=os.path.join(dataset_path,dataset_name), download=True, train=True)
test_data = datasets.MNIST(root=os.path.join(dataset_path,dataset_name),  download=True, train=False)

X_train = np.array(train_data.data).reshape(60000,-1)
Y_train = np.array(train_data.targets).reshape(60000,-1)
X_val = np.array(test_data.data).reshape(10000,-1)[:100]
Y_val = np.array(test_data.targets).reshape(10000,-1)[:100]

# t-SNE降维处理
tsne = TSNE(n_components=3, verbose=1 ,random_state=42)
result = tsne.fit_transform(X_val)
print(tsne.kl_divergence_)

# 归一化处理
scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
result = scaler.fit_transform(result)
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(projection='3d')
ax.set_title('t-SNE process')
ax.scatter(result[:,0], result[:,1], result[:,2] , c=Y_val, s=20)
ax.text(0.5,-0.75,-1.0,s="KL divergence="+str(round(tsne.kl_divergence_,4)))
plt.savefig("t-sne.png")
plt.show()




