#!/usr/bin/env python
# -*- coding: utf-8 -*-
# python main.py --exp_name=carla_dytost_29oct --model=dgcnn --num_points=1024 --k=20
"""

@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: main.py
@Time: 2018/10/13 10:39 PM

"""


from __future__ import print_function

import argparse
import os

import numpy as np
# import sklearn.metrics as metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
# from tqdm import trange
from util import IOStream, cal_loss
from torchsummary import summary
from data import *
import pytorch_lightning as L

# from torchsummary import summary
# from topologylayer.nn import AlphaLayer, BarcodePolyFeature
# from topologylayer.functional.utils_dionysus import *
# from topologylayer.functional.rips_dionysus import Diagramlayer as DiagramlayerRips
# from topologylayer.functional.levelset_dionysus import Diagramlayer as DiagramlayerToplevel





def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')



# width, height = 4,4
# axis_x = np.arange(0, width)
# axis_y = np.arange(0, height)
# grid_axes = np.array(np.meshgrid(axis_x, axis_y))    # 4*4 mesh  
# grid_axes = np.transpose(grid_axes, (1, 2, 0))
# from scipy.spatial import Delaunay
# tri = Delaunay(grid_axes.reshape([-1, 2]))
# faces = tri.simplices.copy()
# F = DiagramlayerToplevel().init_filtration(faces)
# diagramlayerToplevel = DiagramlayerToplevel.apply





class Lightning_DGCNN_module(L.LightningModule):
    def __init__(self, args, output_channels=40):
        super(Lightning_DGCNN_module, self).__init__()
        self.args = args
        self.k = args.k
        
        self.loss_fn = lambda a, b : (a - b).abs().sum(-1).sum(-1).sum(-1)
        self.criterion = self.loss_fn
        
        self.model = DGCNN_Topo(args)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        
        x_hidden2 = get_graph_feature(x1, k=self.k)
        x = self.conv2(x_hidden2)
        x2 = x.max(dim=-1, keepdim=False)[0]
        
        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        
        x_hidden1 = get_graph_feature(x3, k=self.k)   #[batch, x,y]


        x = self.conv4(x_hidden1)
        x4 = x.max(dim=-1, keepdim=False)[0]
        
        x = torch.cat((x1, x2, x3, x4), dim=1)
        
        x = self.conv5(x) # 32, 1024, 1024
        x0 = x.max(dim=-1, keepdim=False)[0]     # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims)
        feat = x0                   # (batch_size, num_points) -> (batch_size, 1, emb_dims)


        x = self.conv6(x)
        
        x = self.conv7(x)
        
        x = self.conv8(x)
        
        x = self.conv9(x)
        
        x = self.conv10(x)
        # x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        # x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        # x = torch.cat((x1, x2), 1)

        # x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        # x = self.dp1(x)
        # x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        # x = self.dp2(x)
        # x = self.linear3(x)
        return x, x_hidden1, x_hidden2, feat
    

    def configure_optimizers(self):
        if self.args.use_sgd:
            print("Use SGD")
            opt = optim.SGD(self.parameters(), lr=self.args.lr*100, momentum=self.args.momentum, weight_decay=1e-4)
        else:
            print("Use Adam")
            opt = optim.Adam(self.parameters(), lr=self.args.lr, weight_decay=1e-4)
        self.scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)
        #the scheduler has to be taken care of in some part
        # scheduler = CosineAnnealingLR(opt, self.args.epochs, eta_min=self.args.lr)
        return opt
    

    def training_step(self, trainloaders, train_idx):

        
        self.scheduler.step()
        k = 0
        train_loss = 0.0
        count = 0.0
        self.model.train()
        train_pred = []
        train_true = []
        # k = 0
        
        dataloader = trainloaders[self.current_epoch%2]
        for data in dataloader:
            # print(k)
            # k+=1
            dynamic_data = data[1]
            static_data = data[2]
            print(dynamic_data.shape, static_data.shape)
            batch_size = dynamic_data.size()[0]
            self.opt.zero_grad()
            # print('going')
            logits, hidden1, hidden2, _ = self.model(dynamic_data)


            # top_loss_out = top_batch_cost(logits.detach().cpu(), diagramlayerToplevel, F)
            # top_loss_hidden1 = top_batch_cost(hidden1.detach().cpu(), diagramlayerToplevel, F)
            # top_loss_hidden2 = top_batch_cost(hidden2.detach().cpu(), diagramlayerToplevel, F)
            loss = self.criterion(logits, static_data)
            # loss += top_loss_out + top_loss_hidden1 + top_loss_hidden2
            loss.backward()
            self.opt.step()
            # preds = logits.max(dim=1)[1]
            count += batch_size
            train_loss += loss.item() * batch_size
            # train_true.append(static_data.cpu().numpy())
            # train_pred.append(preds.detach().cpu().numpy())
    # train_true = np.concatenate(train_true)
    # train_pred = np.concatenate(train_pred)
        outstr = 'Train %d, loss: %.6f' % (self.current_epoch, train_loss*1.0/count)
                                                                                #  metrics.accuracy_score(
                                                                                #      train_true, train_pred),
                                                                                #  metrics.balanced_accuracy_score(
                                                                                #      train_true, train_pred))
        io.cprint(outstr)
        state = {'epoch': self.current_epoch + 1, 'state_dict': self.model.state_dict(),'optimizer': self.opt.state_dict()}
        torch.save(state, f'checkpoints/{self.args.exp_name}/models/model_{self.current_epoch}.t7')


    # def validation_step(self, testloader, batch_idx ):
    #     if self.current_epoch % 3 == 0 :
    #         test_loss = 0.0
    #         count = 0.0
    #         self.model.eval()
    #         test_pred = []
    #         test_true = []
    #         # dataloader = testloaders
            
    #         for data in testloader:
    #             print(static.shape)
    #             # dynamic_data = data[1]
    #             # static_data = data[2]
    #             print(dynamic.shape, static.shape)
    #             batch_size = dynamic_data.size()[0]
    #             logits = self.model(dynamic_data)
    #             loss = self.criterion(logits, static_data)
    #             # preds = logits.max(dim=1)[1]
    #             count += batch_size
    #             test_loss += loss.item() * batch_size
    #             # test_true.append(static_data.cpu().numpy())
    #             # test_pred.append(preds.detach().cpu().numpy())
    #         # test_true = np.concatenate(test_true)
    #         # test_pred = np.concatenate(test_pred)
    #         outstr = 'Test %d, loss: %.6f' % (self.current_epoch, test_loss*1.0/count)
    #                                                                                 #   test_acc,
    #                                                                                 #   avg_per_class_acc)
    #         io.cprint(outstr)
    #         # if test_acc >= best_test_acc:
    #             # best_test_acc = test_acc
    #     state = {'epoch': self.current_epoch + 1, 'state_dict': self.model.state_dict(),'optimizer': self.opt.state_dict()}
    #     torch.save(state, f'checkpoints/{self.args.exp_name}/models/model_{self.current_epoch}.t7')
    #     # torch.save(model.state_dict(), f'checkpoints/{args.exp_name}/models/model_{epoch}.t7')




def main(args):
    _init_()

    # npy = [0,1,2,4,5,6,7,9,10]
    # skip = [6,2,6,1,4,2,2,3,2]
    # headstart = [3,1,3,0,2,1,1,2,1]
    npy = [4]
    skip = [1]
    headstart = [0]
    trainloaders = load_kitti_DyToSt(npy, skip, headstart, args)
    npy = [4]
    skip = [1]
    headstart = [0]
    testloaders = load_kitti_DyToSt(npy, skip, headstart, args)
    #------------------------------------------------------------------------------------
    model = Lightning_DGCNN_module(args)
    print(str(model))
    print('Summary')
    # model = nn.DataParallel(model)
    # summary(model.model, (3,args.beam * int(1024/args.dim)))
    
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    # If use distributed training  PyTorch recommends to use DistributedDataParallel.
    # See: https://pytorch.org/docs/stable/nn.html#torch.nn.DataParallel
    
    trainer = L.Trainer(accelerator="gpu", devices=1, max_epochs=200, log_every_n_steps=5)


    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model, trainloaders)













if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',
                        help='Num of nearest neighbors to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--data', type=str, default='', metavar='N',
                        help='Location of dataset')                 
    parser.add_argument('--mode', type=str, default='carla', required =True,
                        help='Location of dataset')         
    parser.add_argument('--reload', type=str, default='',
                        help='Location of dataset')         
    parser.add_argument('--dim', type=int, default= 4,
                        help='Location of dataset')         
    parser.add_argument('--beam', type=int, default=16,
                        help='Location of dataset')                                                                        
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    torch.manual_seed(args.seed)
    io.cprint(
        'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
    torch.cuda.manual_seed(args.seed)

    main(args)
    
