import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['SemanticKITTI', 'ICCV17ShapeNetSeg', 'EricyiShapeNetSeg', 'Stanford3d'])
parser.add_argument('--model', default='./model-semantickitti.pth')
parser.add_argument('--idx', type=int, default=0, help='model index')
parser.add_argument('--class_choice', type=str, default='', help='class choice')
parser.add_argument('--npoints', default=2500, type=int)
parser.add_argument('--datadir', default='/data')
parser.add_argument('--cache', action='store_true')
parser.add_argument('--topview', action='store_true')
args = parser.parse_args()

from torchinfo import summary
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from time import time
import numpy as np
import pnets as pn
from tqdm import tqdm
import matplotlib.pyplot as plt


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using', device)

aug = pn.aug.Compose(
    pn.aug.Resample(args.npoints),
    pn.aug.Normalize(),
    pn.aug.Jitter(),
    pn.aug.RandomRotation('Z', 0, 2*np.pi),
)
ts = getattr(pn.data, args.dataset)
ts = ts(args.datadir, 'val', None if args.cache else aug)
K = ts.nclasses
if args.cache:
    ts = pn.data.Cache(ts, aug)
#ts = torch.utils.data.Subset(ts, range(10))  # DEBUG
ts = DataLoader(ts, 32, True, num_workers=4, pin_memory=True)

save_path = args.model
model = pn.pointnet.PointNetSeg(K).to(device)
model=torch.load(save_path)
model.eval() 
cmap = plt.cm.get_cmap("hsv", 10)
cmap = np.array([cmap(i) for i in range(10)])[:, :3]
# iterate over val data
for P, Y in tqdm(ts):
        
        #print(P)
        P = P.to(device)
        Y = Y.to(device)
        Y_pred, trans, trans_feat = model(P)
        P_pred = F.softmax(Y_pred, 1)
        K_pred = P_pred.argmax(1).view(-1)
        color=np.empty(K_pred.shape)
        P_pred=P_pred.view(-1,P_pred.shape[1])
        for i in range(len(K_pred)):
            color[i]=P_pred[i,K_pred[i]]  
        P=P.detach().view(P.shape[1],-1) 
        #print(P.shape)
        P=P.cpu().numpy()  
        #print(P[0])'''
        f = pn.plot.plot_topview if args.topview else pn.plot.plot3d
        c = color
        ax = f(P, c)
        pn.plot.zoomin(ax, P)
        ax.set_title(f'class={Y}')
        pn.plot.show()

