import seaborn as sns
from sklearn import tree
# 建模和机器学习
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
import torch
from matplotlib import pyplot as plt
import numpy as np
from sklearn import preprocessing
from sklearn.manifold import TSNE
import os
import warnings
import dtreeviz

warnings.filterwarnings("ignore")


def Local_EX_DT(s_model, train_dataloader, cfg, client_id):
    s_model.eval()
    s_model.to(cfg.device)
    DT_X = []
    DT_Y = []
    count = 0
    with torch.no_grad():
        for x, y in train_dataloader:
            x = x.to(cfg.device)
            _, s_logits = s_model(x)
            # compute loss
            predict_y = torch.argmax(s_logits, dim=-1)
            predict_y = predict_y.cpu().numpy().reshape(-1,1)
            if DT_X == []:
                DT_X = x.cpu().numpy()
            else:
                DT_X = np.vstack((DT_X, x.cpu().numpy()))
            if DT_Y == []:
                DT_Y = predict_y
            else:
                DT_Y = np.vstack((DT_Y,predict_y))
            count += 1
            if count > 1:
                break
    DT_X = DT_X.reshape(DT_X.shape[0],-1)
    DT_Y = DT_Y.reshape(DT_Y.shape[0],-1)

    print("Train DT model...")
    dtModel = DecisionTreeClassifier()  # 建立模型
    dtModel.fit(DT_X, DT_Y)
    # 测试决策树
    prediction = dtModel.predict(DT_X)
    plt.figure(figsize=(10, 7))
    cm = confusion_matrix(DT_Y, prediction)

    ax = sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
    plt.ylabel('Actual label')  # x轴标题
    plt.xlabel('Predicted label')  # y轴标题
    plt.savefig(os.path.join(cfg.logs_dir, f"{client_id}",f"{cfg.dataset}_{cfg.label_data_ratio}_{cfg.distribution_alpha}","DT.png"),bbox_inches='tight', pad_inches=0)
    acc = accuracy_score(DT_Y, prediction)
    print(f"DT fit acc: {acc * 100}")

    # 导出dot
    if cfg.dataset == "mnist":
        classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    elif cfg.dataset == "cifar10":
        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    elif cfg.dataset == "fashionmnist":
        classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    dot_data = tree.export_graphviz(dtModel, out_file=os.path.join(cfg.logs_dir, f"{client_id}",f"{cfg.dataset}_{cfg.label_data_ratio}_{cfg.distribution_alpha}", "DT.dot"),
                                    class_names=classes,
                                    filled=True, rounded=True,
                                    special_characters=True)
    # dot -Tpng breast_cancer_tree_graph.dot -o tree.png


def global_EX_TS(FL_model, test_dataloader, weights_for_clients, cfg, com_round,particpate_ids, is_agg=False):
    for client_id, weight in zip(particpate_ids,weights_for_clients):
        # if client_id % 3 != 0:
        #     continue
        FL_model.load_state_dict(weight)
        FL_model.eval()
        ts_x = []
        ts_y = []
        count = 0
        with torch.no_grad():
            for x, y in test_dataloader:
                x = x.to(cfg.device)
                y = y.to(cfg.device)
                _, x_logits = FL_model(x)
                if ts_x == []:
                    ts_x = x_logits.cpu().numpy()
                else:
                    ts_x = np.vstack((ts_x,x_logits.cpu().numpy()))
                if ts_y == []:
                    ts_y = y.cpu().numpy().reshape(-1,1)
                else:
                    ts_y = np.vstack((ts_y,y.cpu().numpy().reshape(-1,1)))
                count += 1
                if count > 30:
                    break
        # t-SNE降维处理
        tsne = TSNE(n_components=3, verbose=1, random_state=42)
        result = tsne.fit_transform(ts_x)
        # 归一化处理
        scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        result = scaler.fit_transform(result)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        # ax.set_title('t-SNE')
        scatter = ax.scatter(result[:, 0], result[:, 1], result[:, 2], c=ts_y, s=20)
        legend1 = ax.legend(*scatter.legend_elements(), loc="center left", title="Classes")
        ax.text(0.0, -0.75, -1.0, s="KL divergence=" + str(round(tsne.kl_divergence_, 4)),fontsize=16)
        ax.add_artist(legend1)
        if is_agg:
            plt.savefig(os.path.join(cfg.logs_dir, f"{client_id}",f"{cfg.dataset}_{cfg.label_data_ratio}_{cfg.distribution_alpha}",f"{com_round}_TS_agg_after.png"),bbox_inches='tight', pad_inches=0)
        else:
            plt.savefig(os.path.join(cfg.logs_dir, f"{client_id}",f"{cfg.dataset}_{cfg.label_data_ratio}_{cfg.distribution_alpha}", f"{com_round}_TS_agg_before.png"),bbox_inches='tight', pad_inches=0)
        # plt.show()