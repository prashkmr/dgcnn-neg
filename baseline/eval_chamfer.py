import numpy as np
import argparse, os
import logging
from  tqdm import trange
from utils512 import *
import torch
logging.getLogger().setLevel(logging.DEBUG)

parser = argparse.ArgumentParser(description='Arguments for pretain|inversion|eval_treegan|eval_completion.')
parser.add_argument('--location', type=str, default='', help='')
parser.add_argument('--orig', type=str, default='', help='')
parser.add_argument('--dim', type=int, default=4, help='')
parser.add_argument('--beam', type=int, default=64, help='')
args = parser.parse_args()

def getInt(file):
    
    first  = file.find('[')
    second = file.find(']')
    num    = file[first+1:second]
    print(file)
    print(num)
    # exit(0)

def getCD(input, output):
    batch_size = 256
    lossCD =[]
    loss_fn = get_chamfer_dist
    loss  = loss_fn()
    ep    = int(output.shape[0]/batch_size)
    for i in trange(ep):
        lossCD += [( loss( torch.Tensor(input)[i*batch_size:(i+1)*batch_size], torch.Tensor(output)[i*batch_size:(i+1)*batch_size] ) )]
    return torch.stack(lossCD).mean().item()



npy=['s3']

for lidar in npy:
    target = []
    for i in trange(3072):
        file = 'tensor([' + str(i) + '])_target.txt'
        np_output = np.loadtxt(args.location + file, delimiter=';').astype(np.float32).transpose(1,0)
        target.append(np_output)
        
    target = np.array(target)
    orig   =  from_polar_np(np.load(args.orig+lidar +'.npy'))[:,:,::int(16/args.beam),::args.dim][:]
    print(target.shape, orig.shape)
    # exit(0)
    cd = getCD(orig.reshape(-1,3,args.beam, (int(1024/args.dim))), target.reshape(-1,3,args.beam, (int(1024/args.dim))))
    print(cd)


    



