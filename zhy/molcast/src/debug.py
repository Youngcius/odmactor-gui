import torch
import os
from torch import nn, optim
import copy
import torch.nn.functional as F
import numpy as np
import data_process
import model_process
from torch.utils.data import Dataset, DataLoader
from dgl.data import split_dataset
import sys
import random
import argparse
import matplotlib.pyplot as plt
from model_process import  MoleculePredictor, train, test
import pandas as pd
import dgl.function as fn
import time
import pickle

def count_model_parameter(model: nn.Module):
    return sum([p.numel() for p in model.parameters()])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--idx_label', type=int, default=12,
                        help='index choice from 12 labels (default value (10) is the energy U0)')
    parser.add_argument('-dn', '--dim_node', type=int, default=64, help='dim of nodes features after embedded')
    parser.add_argument('-de', '--dim_edge', type=int, default=64, help='dim of edges features after type embedded')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate of Adam optimizer')
    parser.add_argument('-e', '--epochs', type=int, default=6, help='epochs for training')
    # parser.add_argument('-p',action='store_true',default=False,help='Whether use part(20%) of QM9Dataset')
    torch.manual_seed(123)
    np.random.seed(123)
    random.seed(123)
    plt.style.use('seaborn')


    args = parser.parse_args()
    idx_label = args.idx_label
    epochs = args.epochs
    lr = args.lr
    dim_node = args.dim_node
    dim_edge = args.dim_edge

    ##############
    dim_node = 128
    dim_edge = 128
    lr = 5e-5
    epochs = 3
    
    idx_label = 10 # U0
    idx_label = 12 # H
    idx_label = 3 # 偶极矩


    ############
    qm9 = data_process.QM9Dataset()
    

    trainset, evalset, testset = split_dataset(qm9, [0.7, 0.15, 0.15], shuffle=True, random_state=123)
    train_loader = DataLoader(trainset, batch_size=64, shuffle=True, collate_fn=data_process.batcher())
    eval_loader = DataLoader(evalset, batch_size=64, shuffle=True, collate_fn=data_process.batcher())
    test_loader = DataLoader(testset, batch_size=64, shuffle=True, collate_fn=data_process.batcher())

    g = qm9[100][0]
    N = len(qm9)
    dist_extreme = np.array([qm9.get_dist_min_max_with_idx(i) for i in range(N)])
    max_num_atoms = np.max(qm9.dict['num_atoms'])
    dist_min, dist_max = dist_extreme[:, 0].min(), dist_extreme[:, 1].max()
    prop_desc = qm9.get_prop_stat()


    # ### property statistics
    # prop_desc.pop('nobs')
    # for k, v, in prop_desc.items():
    #     print('---'*10)
    #     if type(v) is tuple:
    #         print(k)
    #         print(np.round(v[0][3:]))
    #         print(np.round(v[1][3:]))
    #     else:
    #         print(np.round(v))
    # prop_df = pd.DataFrame(qm9.dict['properties'])
    # prop_df.iloc[:,3:].describe()


    plt.style.use('seaborn')
    model_dir = '../output/'



##############################################
##############################################0
##############################################
##############################################
    lr_adam = 0.0001
    lr_sgd = 0.05
    epochs = 30
    
    for idx_label in [14, 7, 4, 12, 10]:
        print('============={}============'.format(qm9.prop_names[idx_label]))
        with open(os.path.join(model_dir, 'best_model_mae.pkl'),'rb') as f:
            best_model_mae = pickle.load(f)
        model = MoleculePredictor(dim_node=128, dim_edge=128, cutoff_low=np.floor(dist_min),cutoff_high=np.ceil(dist_max),conv_type='dtnn', norm=False,n_conv=4)
        model_name = 'model_label_{}.pkl'.format(idx_label)            
        if os.path.exists(os.path.join(model_dir, model_name)):
            model.load_state_dict(torch.load(os.path.join(model_dir, model_name)))
            print('已从已训练的模型中加载参数！')

        print('Number of parameters:', count_model_parameter(model))
        optimizer = optim.Adam(model.parameters(), lr=lr_adam)
        # optimizer = optim.SGD(model.parameters(), lr=lr_sgd, momentum=0.8)
        # loss_best = 1
        mae_best = best_model_mae[qm9.prop_names[idx_label]]
        best_model_wts = copy.deepcopy(model.state_dict())
        
        running_loss = []
        running_mae = []
        ########################
        # training & evaluating
        for e in range(epochs):
            beg = time.time()
            print('---' * 10)
            print('Epoch: {}'.format(e))
            loss_train, mae_train = train(model, train_loader, optimizer, idx_label=idx_label)
            loss_eval, mae_eval = test(model, test_loader, idx_label=idx_label)
            running_loss.append([loss_train, loss_eval])
            running_mae.append([mae_train, mae_eval])
            # if loss_eval < loss_best:
            if mae_eval < mae_best:
                mae_best = mae_eval
                best_model_wts = copy.deepcopy(model.state_dict())
            end = time.time()
            print('time {} (s) for this epoch; best MAE on eval set: {:.6f}'.format(round(end-beg), mae_best))        
        best_model_mae[qm9.prop_names[idx_label]] = mae_best
        torch.save(best_model_wts, os.path.join(model_dir, model_name))
        print('Best model parameters has been saved!')
        with open(os.path.join(model_dir, 'best_model_mae.pkl'), 'wb') as f:
            pickle.dump(best_model_mae, f)
        print('Best model MAE has been saved!')
                
        running_loss = np.asarray(running_loss)
        running_mae = np.asarray(running_mae)

        # running loss & mae plot
        plt.figure(figsize=(8,4))
        plt.plot(range(10), running_loss[-10:,0], color='k',label='train loss (MSE)')
        plt.plot(range(10), running_loss[-10:,1], color='r', label='eval loss (MSE)')
        plt.plot(range(10), running_mae[-10:,0], '--', color='k',label='train MAE')
        plt.plot(range(10), running_mae[-10:,1], '--', color='r', label='eval MAE')
        plt.legend()
        plt.title('Train/Eval for {} (last 10 epochs)'.format(qm9.prop_names[idx_label]))
        # plt.ylim(0, 10)
        plt.xlabel('Epoch')
        plt.savefig('../output/train_{}'.format(idx_label), dpi=500)
        
        

        ########################
        # testing
        model.load_state_dict(best_model_wts) # use best model to predict
        predict_specific_idx = []
        labels_specific_idx_test = []
        with torch.no_grad():
                for i, (graphs, labels) in enumerate(test_loader):
                    predict_specific_idx += model(graphs).tolist()
                    labels_specific_idx_test += labels[:, idx_label].tolist()

        mse = F.mse_loss(torch.tensor(predict_specific_idx), torch.tensor(labels_specific_idx_test)).item()
        mae = F.l1_loss(torch.tensor(predict_specific_idx), torch.tensor(labels_specific_idx_test)).item()
        print('MSE on test set: {:.4f}'.format(mse))
        print('MAE on test set: {:.4f}'.format(mae)) # 44

        plt.figure(figsize=(8,4))
        plt.hist(labels_specific_idx_test, label='Origin labels', bins=30, alpha=0.5)
        plt.hist(predict_specific_idx, label='Predicted values', bins=30, alpha=0.5)
        plt.legend()
        plt.title('MSE = {:.4f}, MAE = {:.4f}'.format(mse, mae))
        plt.xlabel('Predict: {}'.format(qm9.prop_names[idx_label]))
        plt.ylabel('Count')
        plt.savefig('../output/predict_{}.png'.format(idx_label), dpi=500)





        
    # model = MoleculePredictor(conv_type='mgcn',cutoff_low=np.floor(dist_min),cutoff_high=np.ceil(dist_max))

    # labels_specific_idx = [l[idx_label].item() for _, l in trainset]
    # mean_idx = np.mean(labels_specific_idx)
    # std_idx = np.std(labels_specific_idx)
    # model.set_mean_std(mean_idx, std_idx)
    # print('mean: {}, std: {}'.format(mean_idx, std_idx))
    # model information
    model = MoleculePredictor(conv_type='mgcn',cutoff_low=np.floor(dist_min),cutoff_high=np.ceil(dist_max))

    print(model)
    print('Number of parameters:', count_model_parameter(model))

    running_loss = []
    running_mae = []
    # training & evaluating
    for e in range(3):
        print('---' * 10)
        print('Epoch: {}'.format(e))
        loss_train, mae_train = train(model, train_loader, optimizer, idx_label=idx_label)
        loss_eval, mae_eval = test(model, test_loader, idx_label=idx_label)
        running_loss.append([loss_train, loss_eval])
        running_mae.append([mae_train, mae_eval])
        # if loss_eval < loss_best:
        if mae_eval < mae_best:
            mae_best = mae_eval
            best_model_wts = model.state_dict()
    
    torch.save(best_model_wts, os.path.join(model_dir, model_name))
    print('Best model has been saved!')
    running_loss = np.asarray(running_loss)
    running_mae = np.asarray(running_mae)
    
    # running loss & mae plot
    plt.figure(figsize=(8,4))
    plt.plot(range(epochs), running_loss[:,0], color='k',label='train loss (RMSE)')
    plt.plot(range(epochs), running_loss[:,1], color='r', label='eval loss (RMSE)')
    plt.plot(range(epochs), running_mae[:,0], '--', color='k',label='train MAE')
    plt.plot(range(epochs), running_mae[:,1], '--', color='r', label='eval MAE')
    plt.legend()
    plt.title('Train/Eval for {}'.format(qm9.prop_names[idx_label]))
    plt.ylim(0, 10)
    plt.xlabel('Epoch')
    plt.savefig('../output/train_{}'.format(idx_label), dpi=500)
    
    ####################################
    # 验证集合
    
    loss_test = []
    mae_test = []
    eval_specific_idx = []
    labels_specific_idx_eval = [l[idx_label].item() for _, l in evalset]
    with torch.no_grad():
        for i, (graphs, labels) in enumerate(eval_loader):
            labels = labels[:, idx_label]
            predict = model(graphs)
            eval_specific_idx +=  predict.tolist()
            loss = F.mse_loss(predict, labels)
            mae = F.l1_loss(predict.detach(), labels.detach())
            loss_test.append(loss.item())
            mae_test.append(mae.item())
    
    np.mean(loss_test)
    np.mean(mae_test)
    F.mse_loss(torch.tensor(eval_specific_idx), labels_specific_idx_eval)
    labels_specific_idx_eval[-64:]

    # testing
    predict_specific_idx = []
    labels_specific_idx_test = []
    with torch.no_grad():
            for i, (graphs, labels) in enumerate(test_loader):
                # predict=model(graphs)
                predict_specific_idx += model(graphs).tolist()
                labels_specific_idx_test += labels[:, idx_label].tolist()
    # prop_df.describe()
    mse = F.mse_loss(torch.tensor(predict_specific_idx), torch.tensor(labels_specific_idx_test)).item()
    mae = F.l1_loss(torch.tensor(predict_specific_idx), torch.tensor(labels_specific_idx_test)).item()
    print('MSE on test set: {:.4f}'.format(mse))
    print('MAE on test set: {:.4f}'.format(mae)) # 44


    plt.hist(labels_specific_idx_test, label='Origin labels', bins=30, alpha=0.5)
    plt.hist(predict_specific_idx, label='Predicted values', bins=30, alpha=0.5)
    plt.legend()
    plt.title('MSE = {:.4f}, MAE = {:.4f}'.format(mse, mae))
    plt.xlabel('Predict: {}'.format(qm9.prop_names[idx_label]))
    plt.ylabel('Count')
    plt.savefig('../output/predict_{}.png'.format(idx_label), dpi=400)



