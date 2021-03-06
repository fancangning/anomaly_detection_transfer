import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.nn.init as init
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.utils as v_utils
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm
from torch.utils.tensorboard import SummaryWriter
import cv2
import math
from collections import OrderedDict
import copy
import time

import data.utils as data_utils
import models.loss as loss
import utils
from models import AutoEncoderCov3D, AutoEncoderCov3DMem

import argparse

parser = argparse.ArgumentParser(description="MemoryNormality")
parser.add_argument('--gpu', type=str, default='0', help='which gpu to use')
parser.add_argument('--batch_size', type=int, default=12, help='batch size for training')
parser.add_argument('--epochs', type=int, default=80, help='number of epochs for training')
parser.add_argument('--val_epoch', type=int, default=2, help='evaluate the model every %d epoch')
parser.add_argument('--h', type=int, default=256, help='height of input images')
parser.add_argument('--w', type=int, default=256, help='width of input images')
parser.add_argument('--c', type=int, default=1, help='channel of input images')
parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
parser.add_argument('--t_length', type=int, default=16, help='length of the frame sequences')
parser.add_argument('--ModelName', help='AE/MemAE', type=str, default='MemAE')
parser.add_argument('--ModelSetting', help='Conv3D/Conv3DSpar',type=str, default='Conv3DSpar')  # give the layer details later
parser.add_argument('--MemDim', help='Memory Dimention', type=int, default=2000)
parser.add_argument('--EntropyLossWeight', help='EntropyLossWeight', type=float, default=0.0002)
parser.add_argument('--ShrinkThres', help='ShrinkThres', type=float, default=0.0025)
parser.add_argument('--Suffix', help='Suffix', type=str, default='Non')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for the train loader')
parser.add_argument('--num_workers_test', type=int, default=1, help='number of workers for the test loader')
parser.add_argument('--source_dataset', type=str, default='UCSDped1', help='type of dataset: UCSDped1, UCSDped2, Avenue')
parser.add_argument('--target_dataset', type=str, default='UCSDped2', help='type of dataset: UCSDped1, UCSDped2, Avenue')
parser.add_argument('--dataset_path', type=str, default='./data/', help='directory of data')
parser.add_argument('--exp_dir', type=str, default='log', help='directory of log')
parser.add_argument('--version', type=int, default=0, help='experiment version')

args = parser.parse_args()

torch.manual_seed(2020)

print("--------------PyTorch VERSION:", torch.__version__)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("..............device", device)

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

def arrange_image(im_input):
    im_input = np.transpose(im_input, (0, 2, 1, 3, 4))
    b, t, ch, h, w = im_input.shape
    im_input = np.reshape(im_input, [b * t, ch, h, w])
    return im_input

s_train, s_test = data_utils.give_data_folder(args.source_dataset, args.dataset_path)

print("The training path", s_train)
print("The testing path", s_test)


frame_trans = data_utils.give_frame_trans(args.source_dataset, [args.h, args.w])


s_train_dataset = data_utils.DataLoader(s_train, frame_trans, time_step=args.t_length - 1, num_pred=1)
s_test_dataset = data_utils.DataLoader(s_test, frame_trans, time_step=args.t_length - 1, num_pred=1)

s_train_batch = data.DataLoader(s_train_dataset, batch_size = args.batch_size, 
                              shuffle=True, num_workers=args.num_workers, drop_last=True)
s_test_batch = data.DataLoader(s_test_dataset, batch_size = args.batch_size, 
                             shuffle=False, num_workers=args.num_workers, drop_last=True)

print("Training data shape", len(s_train_batch))
print("Validation data shape", len(s_test_batch))

# Model setting

if (args.ModelName == 'AE'):
    model = AutoEncoderCov3D(args.c)
elif(args.ModelName=='MemAE'):
    model = AutoEncoderCov3DMem(args.c, args.MemDim, shrink_thres=args.ShrinkThres)
else:
    model = []
    print('Wrong Name.')

    
model = model.to(device)
parameter_list = [p for p in model.parameters() if p.requires_grad]

for name, p in model.named_parameters():
    if not p.requires_grad:
        print("---------NO GRADIENT-----", name)
        
optimizer = torch.optim.Adam(parameter_list, lr = args.lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.2)  # version 2

#scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max =args.epochs)

# Report the training process
log_dir = os.path.join(args.exp_dir, args.source_dataset+'_to_'+args.target_dataset, 'lr_%.5f_entropyloss_%.5f_version_%d' % (
    args.lr, args.EntropyLossWeight, args.version))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
orig_stdout = sys.stdout
f = open(os.path.join(log_dir, 'log.txt'),'w')
sys.stdout= f

for arg in vars(args):
    print(arg, getattr(args, arg))

train_writer = SummaryWriter(log_dir=log_dir)

# warmup
model.train()
with torch.no_grad():
    for batch_idx, frame in enumerate(s_train_batch):
        frame = frame.reshape([args.batch_size, args.t_length, args.c, args.h, args.w])
        frame = frame.permute(0, 2, 1, 3, 4)
        frame = frame.to(device)
        model_output = model(frame)

# Training
for epoch in range(args.epochs):
    model.train()
    tr_re_loss, tr_mem_loss, tr_tot = 0.0, 0.0, 0.0
    progress_bar = tqdm(s_train_batch)

    for batch_idx, frame in enumerate(progress_bar):
        progress_bar.update()
        frame = frame.reshape([args.batch_size, args.t_length, args.c, args.h, args.w])
        frame = frame.permute(0, 2, 1, 3, 4)
        frame = frame.to(device)
        optimizer.zero_grad()

        model_output = model(frame)
        recons, attr = model_output['output'], model_output['att']
        re_loss = loss.get_reconstruction_loss(frame, recons, mean=0.5, std=0.5)
        mem_loss = loss.get_memory_loss(attr)
        tot_loss = re_loss + mem_loss * args.EntropyLossWeight
        tr_re_loss += re_loss.data.item()
        tr_mem_loss += mem_loss.data.item()
        tr_tot += tot_loss.data.item()
        
        tot_loss.backward()
        optimizer.step()
        
    train_writer.add_scalar("model/train-recons-loss", tr_re_loss/len(s_train_batch), epoch)
    train_writer.add_scalar("model/train-memory-sparse", tr_mem_loss/len(s_train_batch), epoch)
    train_writer.add_scalar("model/train-total-loss", tr_tot/len(s_train_batch), epoch)
    scheduler.step()
    
    current_lr = optimizer.param_groups[0]['lr']
    train_writer.add_scalar('learning_rate', current_lr, epoch)
    
    if epoch % args.val_epoch == 0:
        with torch.no_grad():
            model.eval()
            re_loss_val, mem_loss_val = 0.0, 0.0
            for batch_idx, frame in enumerate(s_test_batch):
                frame = frame.reshape([args.batch_size, args.t_length, args.c, args.h, args.w])
                frame = frame.permute(0, 2, 1, 3, 4)
                frame = frame.to(device)
                model_output = model(frame)
                recons, attr = model_output['output'], model_output['att']
                re_loss = loss.get_reconstruction_loss(frame, recons, mean=0.5, std=0.5)
                mem_loss = loss.get_memory_loss(attr)
                re_loss_val += re_loss.data.item()
                mem_loss_val += mem_loss.data.item()
                if batch_idx == len(s_test_batch) - 1:
                    _input_npy = frame.detach().cpu().numpy()
                    _input_npy = _input_npy * 0.5 + 0.5
                    _recons_npy = recons.detach().cpu().numpy()
                    _recons_npy = _recons_npy * 0.5 + 0.5  # [batch_size, ch, time, imh, imw]
                    train_writer.add_images("image/input_image", arrange_image(_input_npy), epoch)
                    train_writer.add_images("image/reconstruction", arrange_image(_recons_npy), epoch)
            train_writer.add_scalar("model/val-recons-loss", re_loss_val / len(s_test_batch), epoch)
            train_writer.add_scalar("model/val-memory-sparse", mem_loss_val / len(s_test_batch), epoch)
            print("epoch %d" % epoch, "recons loss training %.4f validation %.4f" % (tr_re_loss, re_loss_val), 
                  "memory sparsity training %.4f validation %.4f" % (tr_mem_loss, mem_loss_val))

    if epoch >= args.epochs - 50:
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            torch.save(model.state_dict(), log_dir + "/model-{:04d}.pt".format(epoch))

sys.stdout = orig_stdout
f.close()




