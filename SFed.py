############### test KDHT ##############
import copy
import torch
from torch import nn
import os
from torch.nn import functional as F
import warnings
warnings.filterwarnings("ignore")
from torchmetrics import MetricTracker, Accuracy, Recall, Precision, Specificity, ConfusionMatrix, F1Score
from XFed import Local_EX_DT
import json
from neural_nets import CAE
import time
import numpy as np
import torchvision

def show_images(pred_images, fname):
    imgs_sample = (pred_images.data + 1) / 2.0
    filename = os.path.join(fname, "rec_x.jpg")
    torchvision.utils.save_image(imgs_sample, filename, nrow=10)

# training based on KDHT
def local_train(s_model,t_model, train_dataloader,test_dataloader, cfg, client_id, com_round, temp = 3):
    start = time.time()
    os.makedirs(os.path.join(cfg.logs_dir, f"{client_id}", f"{cfg.dataset}_{cfg.label_data_ratio}_{cfg.distribution_alpha}"),exist_ok=True)
    checkpoint_path = os.path.join(cfg.checkpoints_dir, f"client_{client_id}")
    # laod weights
    s_weights_path = os.path.join(checkpoint_path,"s.pth")
    weights = torch.load(s_weights_path, map_location="cpu")
    s_model.load_state_dict(weights)
    t_weights_path = os.path.join(checkpoint_path, "t.pth")
    weights = torch.load(t_weights_path, map_location="cpu")
    t_model.load_state_dict(weights)

    s_model = s_model.to(cfg.device)
    t_model = t_model.to(cfg.device)
    # define optimizer

    optimizer_s = torch.optim.Adam(s_model.parameters(), lr=cfg.learning_rate,
                                             weight_decay=cfg.weight_delay)
    optimizer_t = torch.optim.Adam(t_model.parameters(), lr=cfg.learning_rate,
                                   weight_decay=cfg.weight_delay)

    # optimizer_s = torch.optim.SGD(s_model.parameters(), lr=cfg.learning_rate,
    #                                          weight_decay=cfg.weight_delay,momentum=0.9)
    # scheduler_s = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_s, mode='min',
    #                                                                   factor=0.1, patience=50,
    #                                                                   verbose=True, threshold=0.0001,
    #                                                                   threshold_mode='rel',
    #                                                                   cooldown=0, min_lr=0, eps=1e-08)



    # define loss function
    # mse = nn.MSELoss()
    kl = nn.KLDivLoss()
    # kl = nn.KLDivLoss(reduction="batchmean")  # 不包含softmax操作(所以可以自己设定温度系数)
    # training
    test_metrics = {"Loss": [], "F1": [], "Acc": [], "Spc": [], "Rcl": [], "Pcl": [], "Conf": []}
    t_model.train()
    s_model.train()
    data_size = train_dataloader.__len__()
    label_size = int(data_size*cfg.label_data_ratio)
    # unsupervised learning
    for epoch in range(cfg.epochs_for_clients):
        for i, (x, y) in enumerate(train_dataloader):
            if i < label_size:
                continue
            optimizer_t.zero_grad()
            x = x.to(cfg.device)
            # unlabel
            t_latent, x_ = t_model(x)
            # compute l1 loss
            loss = F.l1_loss(x_,x)
            loss.backward()
            optimizer_t.step()
            # scheduler_s.step(loss)
        if epoch == cfg.epochs_for_clients - 1:
            x_ = x_.detach().cpu()
            show_images(x_, os.path.join(cfg.logs_dir, f"{client_id}",
                                         f"{cfg.dataset}_{cfg.label_data_ratio}_{cfg.distribution_alpha}"))
        print(f"client{client_id}-epoch {epoch} | CAE loss: {loss.item()}")
    torch.save(t_model.state_dict(), t_weights_path)
    cae_loss = loss.item()
    # train CNN
    t_model.eval()
    for epoch in range(cfg.epochs_for_clients):
        for i, (x, y) in enumerate(train_dataloader):
            if i > label_size:
                break
            optimizer_s.zero_grad()
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            s_latent, cls = s_model(x)
            with torch.no_grad():
                t_latent, x_ = t_model(x)
            # compute div loss
            s_latent_softmax = F.log_softmax(s_latent / (temp/cae_loss), dim=1)
            t_latent_softmax = F.softmax(t_latent / (temp/cae_loss), dim=1)
            kl_loss = kl(s_latent_softmax, t_latent_softmax) * 1e3
            loss = F.cross_entropy(cls,y) + kl_loss
            loss.backward()
            optimizer_s.step()
        print(f"client{client_id}-epoch {epoch} | CNN loss: {loss.item()}")
    # test
    s_model.eval()
    for i, (x, y) in enumerate(train_dataloader):
        x = x.to(cfg.device)
        y = y.to(cfg.device)
        with torch.no_grad():
            s_latent, cls = s_model(x)
        F1_score, ACC, Spc, Rcl, Pcl, Conf = calculate_metrics(torch.argmax(cls, dim=-1), y,
                                                                    num_class=cfg.num_class)
        test_metrics["F1"].append(F1_score)
        test_metrics["Acc"].append(ACC)
        test_metrics["Spc"].append(Spc)
        test_metrics["Rcl"].append(Rcl)
        test_metrics["Pcl"].append(Pcl)
        test_metrics["Conf"].append(Conf)
        test_metrics["Loss"].append(loss.item())
    # save logs
    test_metrics["F1"] = float(np.mean(test_metrics["F1"]))
    test_metrics["Acc"] = float(np.mean(test_metrics["Acc"]))
    test_metrics["Spc"] = float(np.mean(test_metrics["Spc"]))
    test_metrics["Rcl"] = float(np.mean(test_metrics["Rcl"]))
    test_metrics["Pcl"] = float(np.mean(test_metrics["Pcl"]))
    test_metrics["Loss"] = float(np.mean(test_metrics["Loss"]))
    test_metrics["Conf"] = test_metrics["Conf"]
    print(f"round {com_round},client {client_id}:",test_metrics)
    with open(os.path.join(cfg.logs_dir, f"{client_id}", f"{cfg.dataset}_{cfg.label_data_ratio}_{cfg.distribution_alpha}", f"round_{com_round}.json"), "w",
              encoding="utf-8")as f:
        f.write(json.dumps(test_metrics, ensure_ascii=False, indent=4))
    print("Train DT for local explaining...")
    # local exp
    if cfg.use_ulex:
        Local_EX_DT(s_model, train_dataloader, cfg, client_id)

    print("waste time:", time.time() - start)
    return copy.deepcopy(s_model.state_dict())

def calculate_metrics(preds,targets,num_class=10,device="cuda"):
    preds = preds.to(device)
    targets = targets.to(device)
    test_f1 = F1Score(num_classes=num_class, threshold=1. / 5, average="macro",task="multiclass").to(device)  # F1 score
    test_acc = Accuracy(num_classes=num_class, threshold=1. / 5, average="micro",task="multiclass").to(device) # Accuracy
    test_rcl = Recall(num_classes=num_class, threshold=1. / 5, average="macro",task="multiclass").to(device)  # Recall
    test_pcl = Precision(num_classes=num_class, threshold=1. / 5, average="macro",task="multiclass").to(device)  # Precision
    test_spc = Specificity(num_classes=num_class, threshold=1. / 5, average="macro",task="multiclass").to(device)  # Specificity
    test_conf_mat = ConfusionMatrix(num_classes=num_class, threshold=1. / 5,task="multiclass").to(device)  # Confusion Matrix

    F1_score = test_f1(preds, targets).cpu().numpy()
    ACC = test_acc(preds, targets).cpu().numpy()
    Rcl = test_rcl(preds, targets).cpu().numpy()
    Pcl = test_pcl(preds, targets).cpu().numpy()
    Spc = test_spc(preds, targets).cpu().numpy()
    Conf = test_conf_mat(preds, targets).cpu().numpy().tolist()
    return F1_score,ACC,Spc,Rcl,Pcl,Conf


if __name__ == '__main__':
    H1 = torch.rand((5,5))
    H2 = torch.rand((5,5))
    print(H1)
    H1 = F.log_softmax(H1,dim=0)
    H2 = F.softmax(H2,dim=0)
    dis = F.kl_div(H1,H2)
    print("dis:",dis)














