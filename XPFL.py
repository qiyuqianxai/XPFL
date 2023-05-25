import torch
from torch import nn
import numpy as np
import json
from torch.nn import functional as F
from Config import Config
import random
from torchsummary import summary
import copy
import os
import neural_nets
from get_data import get_data_loaders
from SFed import local_train
import time
from XFed import global_EX_TS

def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def initial_global_model(cfg, test_dataloader, global_model):
    # initial global model on server
    global_model.to(cfg.device)
    global_model.train()
    start = time.time()
    # define optimizer
    optimizer = torch.optim.Adam(global_model.parameters(), lr=cfg.learning_rate,
                                             weight_decay=cfg.weight_delay)
    # optimizer = torch.optim.SGD(global_model.parameters(), lr=cfg.learning_rate,
    #                              weight_decay=cfg.weight_delay,momentum=0.9)
    # define loss function
    crossentropy = nn.CrossEntropyLoss()
    # training
    for epoch in range(1):
        for x, y in test_dataloader:
            optimizer.zero_grad()
            x = x.to(cfg.device)
            y = y.to(cfg.device)
            _,logits = global_model(x)
            loss = crossentropy(logits, y)
            acc = (torch.argmax(logits, dim=-1) == y).float().mean()
            loss.backward()
            optimizer.step()
        print(f"Initial global model, epoch,{epoch},acc:{acc},loss:{loss.item()}")
    print("waste time:", time.time() - start)

def meta_update_model(cfg, FL_model, wavg, weights_for_clients,test_dataloader):
    FL_model.load_state_dict(wavg)
    FL_model.to(cfg.device)
    crossentropy = nn.CrossEntropyLoss()
    for x, y in test_dataloader:
        x = x.to(cfg.device)
        y = y.to(cfg.device)
        _, logits = FL_model(x)
        loss = crossentropy(logits, y)
        loss.backward()
    # meta updation
    temp_model = copy.deepcopy(FL_model)
    new_weights_for_clients = []
    for client_weight in weights_for_clients:
        temp_model.load_state_dict(client_weight)
        cl_opt = torch.optim.Adam(temp_model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_delay)
        cl_opt.zero_grad()
        for para1,para2 in zip(temp_model.parameters(),FL_model.parameters()):
            para1.grad = para2.grad
        cl_opt.step()
        new_weights_for_clients.append(temp_model.state_dict())
    return new_weights_for_clients

def sl_update_model(cfg, FL_model, wavg, weights_for_clients, test_dataloader):
    FL_model.load_state_dict(wavg)
    FL_model.to(cfg.device)
    crossentropy = nn.CrossEntropyLoss()
    gl_opt = torch.optim.Adam(FL_model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_delay)
    FL_model.train()
    for x, y in test_dataloader:
        gl_opt.zero_grad()
        x = x.to(cfg.device)
        y = y.to(cfg.device)
        _, logits = FL_model(x)
        loss = crossentropy(logits, y)
        loss.backward()
        gl_opt.step()
    global_weight = FL_model.state_dict()
    for cl_weight in weights_for_clients:
        for key in cl_weight.keys():
            para1 = cl_weight[key]
            para2 = global_weight[key]
            l2 = F.mse_loss(para1,para2)
            cl_weight[key] = para2 + para1*l2
    return weights_for_clients

def EXFL(clients_loader, train_dataloader, test_dataloader, cfg):
    if cfg.dataset == "cifar10":
        s_model = getattr(neural_nets, "CNN")(3)
        t_model = getattr(neural_nets, "CAE")(3)
    else:
        s_model = getattr(neural_nets, "CNN")(1)
        t_model = getattr(neural_nets, "CAE")(1)

    # initial_global_model(cfg,test_dataloader,s_model)
    # broadcast
    print("broadcast weights to clients")
    for i in range(cfg.n_clients):
        checkpoint_path = os.path.join(cfg.checkpoints_dir, f"client_{i}")
        os.makedirs(checkpoint_path, exist_ok=True)
        torch.save(s_model.state_dict(), os.path.join(checkpoint_path,"s.pth"))
        torch.save(t_model.state_dict(), os.path.join(checkpoint_path,"t.pth"))
    for com_round in range(cfg.communication_rounds):
        weights_for_clients = []
        # local training
        particpate_ids = []
        for client_id in range(cfg.n_clients):
            error_prob = random.random()
            if error_prob < cfg.error_ratio:
                continue
            s_weights = local_train(s_model, t_model, clients_loader[client_id],test_dataloader, cfg, client_id, com_round)
            weights_for_clients.append(s_weights)
            particpate_ids.append(client_id)
        print(particpate_ids)

        # server do:
        # global visual before update
        if cfg.use_ulex and (com_round%10 == 0 or com_round==cfg.communication_rounds-1):
            global_EX_TS(s_model, test_dataloader, weights_for_clients, cfg, com_round,particpate_ids, False)
        print("merge weights")
        w_avg = copy.deepcopy(weights_for_clients[0])
        for k in w_avg.keys():
            if "num_batches_tracked" in k:
                continue
            w_avg[k] = w_avg[k] * clients_loader[0].dataset.__len__()
            for i in range(1,len(weights_for_clients)):
                w_avg[k] += weights_for_clients[i][k] * clients_loader[i].dataset.__len__()
            w_avg[k] = torch.div(w_avg[k],train_dataloader.dataset.__len__())
        for i,cl_weight in enumerate(weights_for_clients):
            for key in cl_weight.keys():
                para1 = cl_weight[key]
                para2 = w_avg[key]
                para1_flatten = para1.view(1,-1)
                para2_flatten = para2.view(1,-1)
                dis = F.cosine_similarity(para1_flatten, para2_flatten,dim=1)
                cl_weight[key] = para1*dis + para2*(1-dis)
        # weights_for_clients = sl_update_model(cfg, s_model, w_avg, weights_for_clients, test_dataloader)

        # global visual after update
        if cfg.use_ulex and (com_round%10==0 or com_round==cfg.communication_rounds-1):
            global_EX_TS(s_model,test_dataloader,weights_for_clients,cfg, com_round, particpate_ids, True)

        print("broadcast weights to clients")
        for i, weight in enumerate(weights_for_clients):
            # updated s model as the new t model
            checkpoint_path = os.path.join(cfg.checkpoints_dir, f"client_{i}")
            os.makedirs(checkpoint_path, exist_ok=True)
            torch.save(weight, os.path.join(checkpoint_path, "s.pth"))

if __name__ == '__main__':
    # hyparametes set
    torch.cuda.set_device(0)
    same_seeds(2048)
    cfg = Config()
    # prepare data
    client_loaders, train_loader, test_loader = get_data_loaders(cfg, True)
    EXFL(client_loaders, train_loader, test_loader, cfg)
    # for ratio in [0.2]:
    #     cfg.label_data_ratio = ratio
    #     for alpha in [0.3,0.5,1]:
    #         cfg.distribution_alpha = alpha
    #         print(cfg.label_data_ratio)
    #         EXFL(client_loaders, train_loader, test_loader, cfg)





