import torch
from torch import nn
import os
import sys
import numpy as np
import torchvision
import time
from torchvision import transforms
from torch.utils.data import DataLoader
import utils
from models import AutoEncoderCov3D, AutoEncoderCov3DMem, AdversarialAutoEncoderCov3DMem
import data.utils as data_utils
import argparse
from tqdm import tqdm
import utils.eval as eval_utils
import models.loss as loss
from sklearn import metrics
import matplotlib.pyplot as plt
import time

parser = argparse.ArgumentParser(description="Memorizing_Normality")
parser.add_argument('--gpu', type=str, default='0', help='the gpu to use')
parser.add_argument('--source_dataset', type=str, default='UCSDped1')
parser.add_argument('--target_dataset', type=str, default="UCSDped2")
parser.add_argument("--dataset_path", type=str, default='./data')
parser.add_argument("--version", type=int, default=0)
parser.add_argument("--model_name", type=str, default='model-pretrain.pt')
parser.add_argument("--EntropyLossWeight", type=float, default=0.0002)
parser.add_argument('--AdversarialLossWeight', help='AdversarialLossWeight', type=float, default=1e-4)
parser.add_argument("--lr", type=float, default=5e-3)
parser.add_argument("--exp_dir", type=str, default='log')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
height, width = 256, 256
ch = 1
num_frame = 16
batch_size = 1
ModelName = "AdversarialAE"
model_dir = os.path.join(args.exp_dir, args.source_dataset + 'to' + args.target_dataset + '_' + ModelName,
                         'lr_%.5f_AdversarialLossWeight_%.5f_version_%d' % (args.lr, args.AdversarialLossWeight, args.version))

orig_stdout = sys.stdout
f = open(os.path.join(model_dir, 'output_%s_%s.txt' % ("original_1.00", args.model_name)), 'w')
sys.stdout = f

ckpt_dir = os.path.join(model_dir, args.model_name)

gt_file = "ckpt/%s_gt.npy" % (args.target_dataset)
# gt_file = args.dataset_path + "%s/gt_label.npy" % args.dataset_type
save_path = os.path.join(model_dir, "recons_error_original_1.0_%s.npy" % args.model_name)

# if os.path.isfile(save_path):
#     recons_error = np.load(save_path)
#     eval_utils.eval_video2(gt_file, recons_error, args.dataset_type)
#     exit()

if args.target_dataset == "Avenue":
    data_dir = os.path.join(args.dataset_path, "Avenue/frames/testing/")
elif "UCSD" in args.target_dataset:
    data_dir = os.path.join(args.dataset_path, "%s/Test_jpg/" % args.target_dataset)
else:
    raise Exception("The dataset is not available..........")

frame_trans = transforms.Compose([
    transforms.Resize([height, width]),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])
unorm_trans = utils.UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

print("------Model folder", model_dir)
print("------Restored ckpt", ckpt_dir)

label = np.load(gt_file, allow_pickle=True)
# ped2toped1 20 ped1toped2 1
data_loader = data_utils.DataLoader(data_dir, frame_trans, label, False, time_step=num_frame - 1, num_pred=1, video_start=1, video_end=2)
video_data_loader = DataLoader(data_loader, batch_size=batch_size, shuffle=False)

chnum_in = 1
mem_dim_in = 2000
sparse_shrink_thres = 0.0025

model = AdversarialAutoEncoderCov3DMem(chnum_in, backward_coeff=0.0, mem_dim=mem_dim_in, shrink_thres=sparse_shrink_thres)
model_para = torch.load(ckpt_dir)
model.load_state_dict(model_para)
model.requires_grad_(False)
model.to(device)
model.eval()

img_crop_size = 0
recon_error_list = list()
label_list = list()
# recon_error_list = []
start = time.clock()
idx = 0
for frames, label in video_data_loader:
    frames = frames.reshape([batch_size, num_frame, ch, height, width])
    frames = frames.permute(0, 2, 1, 3, 4)
    frames = frames.to(device)
    if (ModelName == 'AE'):
        recon_frames = model(frames)
        ###### calculate reconstruction error (MSE)
        recon_np = utils.vframes2imgs(unorm_trans(recon_frames.data), step=1, batch_idx=0)
        input_np = utils.vframes2imgs(unorm_trans(frames.data), step=1, batch_idx=0)
        r = utils.crop_image(recon_np, img_crop_size) - utils.crop_image(input_np, img_crop_size)
        # recon_error = np.mean(sum(r**2)**0.5)
        recon_error = np.mean(r ** 2)  # **0.5
    elif (ModelName == 'MemAE'):
        recon_res = model(frames)
        recon_frames = recon_res['output']
        recon_np = utils.vframes2imgs(unorm_trans(recon_frames.data), step=1, batch_idx=0)
        input_np = utils.vframes2imgs(unorm_trans(frames.data), step=1, batch_idx=0)
        r = utils.crop_image(recon_np, img_crop_size) - utils.crop_image(input_np, img_crop_size)
        sp_error_map = sum(r ** 2) ** 0.5
        recon_error = np.mean(sp_error_map.flatten())
    elif ModelName == 'AdversarialAE':
        recon_res = model(frames)
        recon_frames = recon_res['output']
#         torchvision.utils.save_image((torch.abs(recon_frames[0, 0, 8, :, :]-frames[0, 0, 8, :, :])).cpu(), './'+args.model_name.split('.')[0]+'/%d.png'%idx, normalize=True)
#         torchvision.utils.save_image(recon_frames[0, 0, 8, :, :].cpu(), './'+args.model_name.split('.')[0]+'/recon_%d.png'%idx, normalize=True)
#         torchvision.utils.save_image(frames[0, 0, 8, :, :].cpu(), './'+args.model_name.split('.')[0]+'/original_%d.png'%idx, normalize=True)
        _, re_loss_per_sample = loss.get_reconstruction_loss(frames, recon_frames, mean=0.5, std=0.5)
        recon_error_list.append(re_loss_per_sample)
        label_list.append(label)
#         print('recon_loss_per_sample:')
#         print(re_loss_per_sample)
#         print('label:')
#         print(label)
    else:
        recon_error = -1
        print('Wrong ModelName.')
    idx += 1
end = time.clock()
print(end-start)

recon_error_list = torch.cat(recon_error_list, 0)
label = torch.cat(label_list, 0)

recon_error_list = np.array(recon_error_list.detach().cpu())
label = np.array(label.detach().cpu())

# print('recon_error_list:')
# print(recon_error_list)
# print('label:')
# print(label)
# print('---------------------------------------------------')

fpr, tpr, thresholds = metrics.roc_curve(np.array(label), np.array(recon_error_list))
auc = metrics.auc(fpr, tpr)

# plot
fig, ax = plt.subplots()
plt.plot(fpr, tpr, label='ROC(area = {0:.4f})'.format(auc), lw=2)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.savefig(os.path.join(model_dir, args.source_dataset+'to'+args.target_dataset+'_'+args.model_name[:-3]+'_roc_curve.png'))

sys.stdout = orig_stdout
np.save(save_path, recon_error_list)
f.close()
