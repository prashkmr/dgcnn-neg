from __future__ import print_function
import torch.utils.data as data
from torch.utils.data import Dataset
import os
import os.path
import torch
import numpy as np
import h5py
import random
import numpy as np
from tqdm import trange


class CRNShapeNet(data.Dataset):
    """
    Dataset with GT and partial shapes provided by CRN
    Used for shape completion and pre-training tree-GAN
    """
    def __init__(self, args):
        self.args = args
        self.dataset_path = self.args.dataset_path
        self.class_choice = self.args.class_choice
        self.split = self.args.split

        pathname = os.path.join(self.dataset_path, f'{self.split}_data.h5')
        
        data = h5py.File(pathname, 'r')
        self.gt = data['complete_pcds'][()]
        self.partial = data['incomplete_pcds'][()]
        self.labels = data['labels'][()]
        
        np.random.seed(0)
        cat_ordered_list = ['plane','cabinet','car','chair','lamp','couch','table','watercraft']

        cat_id = cat_ordered_list.index(self.class_choice.lower())
        self.index_list = np.array([i for (i, j) in enumerate(self.labels) if j == cat_id ])                      

    def __getitem__(self, index):
        full_idx = self.index_list[index]
        gt = torch.from_numpy(self.gt[full_idx]) # fast alr
        label = self.labels[index]
        partial = torch.from_numpy(self.partial[full_idx])
        return gt, partial, full_idx

    def __len__(self):
        return len(self.index_list)


class KITTI_loader(data.Dataset):
    """
    Dataset of numbers in [a,b] inclusive
    """

    def __init__(self, args):
        super(KITTI_loader, self).__init__()
        self.args = args
        self.dataset_path = self.args.dataset_path # the folder
        static = []
        dynamic = []
        npylist = [8,9,10,11,12,13,14,15]
        for data_npy in npylist:
            
            npyfile = np.load(self.dataset_path + "s" + str(data_npy) + '.npy')
            dataxyz = self.from_polar_np(npyfile)
            # print(f"dataxyz shape {dataxyz.shape}") 
            datai = dataxyz[:,:,::int(64/args.beam),::args.dim].transpose(0,2,3,1).reshape(-1, args.beam * int(512/args.dim), 3)
            print('Orig', dataxyz.shape)
            print('New', datai.shape)
            static.append(datai)

            npyfile = np.load(self.dataset_path + "d" + str(data_npy) + '.npy')
            dataxyz = self.from_polar_np(npyfile)
            # print(f"dataxyz shape {dataxyz.shape}") 
            datai = dataxyz[:,:,::int(64/args.beam),::args.dim].transpose(0,2,3,1).reshape(-1, args.beam * int(512/args.dim), 3)
            print('Orig', dataxyz.shape)
            print('New', datai.shape)
            dynamic.append(datai) 


        self.static = np.concatenate(static, axis=0)
        self.dynamic = np.concatenate(dynamic, axis=0)
        # self.complete = np.concatenate([self.from_polar_np(np.load(self.dataset_path, mmap_mode='r')[:, :, :, ::8]) for i in range(2)], axis=0).transpose(0, 2, 3, 1).reshape(-1, 2048, 3)
        # print(self.dataset.shape)

    def __len__(self):
        # batchsize
        return self.static.shape[0]

    def __getitem__(self, index):
        
        return self.static[index], self.dynamic[index], int(index)

    def from_polar_np(self, velo):
        angles = np.linspace(0, np.pi * 2, velo.shape[-1])
        dist, z = velo[:, 0], velo[:, 1]
        x = np.cos(angles) * dist
        y = np.sin(angles) * dist
        out = np.stack([x,y,z], axis=1)
        return out.astype('float32')







# Attention loader for KITTI in the form of alternating between loaders for KITTI dataset-----------------------------------------------------------------------------------------------------------------------------------


class Attention_loader_dytost(Dataset):
    """
    Dataset of numbers in [a,b] inclusive
    """

    def __init__(self, dynamic, static):
        super(Attention_loader_dytost, self).__init__()

        self.dynamic = dynamic
        self.static = static

    def __len__(self):
        return self.dynamic.shape[0]

    def __getitem__(self, index):
        
        return  self.static[index], self.dynamic[index], index



def load_kitti_DyToSt(npy, skip ,headstart, args):
    # npy = [str(i) for i in range(10)]
    # npy.remove('8')
    # skip = [6,2,6,2,1,4,2,2,3,2]
    # headstart = [3,1,3,1,0,2,1,1,2,1]
    st1 = []
    dy1 = []
    st2 = []
    dy2 = []
    commonSt = []
    commonDy = []

    for i in trange(len(npy)):
        st = np.load(args.dataset_path + 'static/' + str(npy[i])+ '.npy')[:,:,::int(64/args.beam),::args.dim].astype('float32').transpose(0,2,3,1).reshape(-1, args.beam * int(1024/args.dim), 3)
        dy = np.load(args.dataset_path + 'dynamic/' + str(npy[i])+ '.npy')[:,:,::int(64/args.beam),::args.dim].astype('float32').transpose(0,2,3,1).reshape(-1, args.beam * int(1024/args.dim), 3)
        # print(st.shape, dy.shape)   #correct
        
        st1.append(st[::skip[i]])
        dy1.append(dy[::skip[i]])
        
        st2.append(st[headstart[i]:][::skip[i]])
        dy2.append(dy[headstart[i]:][::skip[i]])

    st1 = np.concatenate(st1, axis=0)
    dy1 = np.concatenate(dy1, axis=0)
    st2 = np.concatenate(st2, axis=0)
    dy2 = np.concatenate(dy2, axis=0)
    print(st1.shape, dy1.shape, st2.shape, dy2.shape)


    data1 = Attention_loader_dytost(dy1, st1)
    data2 = Attention_loader_dytost(dy2, st2)
    loader1  = torch.utils.data.DataLoader(data1, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
    loader2  = torch.utils.data.DataLoader(data2, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    return [loader1, loader2]


# Attention loader for KITTI in the form of alternating between loaders for KITTI dataset-----------------------------------------------------------------------------------------------------------------------------------


