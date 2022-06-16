'''
Plots a given dataset.

Usage examples:
$ python3 example_plot.py SemanticKITTI 0
$ python3 example_plot.py SemanticKITTI 0 --topview
'''
import pnets as pn
from torchinfo import summary
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('dataset', choices=['Sydney', 'SemanticKITTI', 'ICCV17ShapeNetSeg', 'ICCV17ShapeNetClass', 'EricyiShapeNetSeg', 'EricyiShapeNetClass', 'Stanford3d'])
parser.add_argument('i', type=int)
parser.add_argument('--datadir', default='/data')
parser.add_argument('--topview', action='store_true')
parser.add_argument('--model', default='./model-semantickitti.pth')

args = parser.parse_args()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using', device)

ds = getattr(pn.data, args.dataset)
ds = ds(args.datadir, 'val', None)
K = ds.nclasses
P, Y = ds[args.i]
P,Y = DataLoader([P,Y], 1, True, num_workers=4, pin_memory=True)
P = P.to(device)
Y = Y.to(device)

save_path = args.model
model = pn.pointnet.PointNetSeg(K).to(device)
model=torch.load(save_path)
model.eval() 

Y_pred, trans, trans_feat = model(P)
P_pred = F.softmax(Y_pred, 1)
K_pred = P_pred.argmax(1)
f = pn.plot.plot_topview if args.topview else pn.plot.plot3d
c = Y if ds.segmentation else 'k'
ax = f(K_pred, c)
pn.plot.zoomin(ax, P)
if not ds.segmentation:
    ax.set_title(f'class={Y}')
pn.plot.show()
