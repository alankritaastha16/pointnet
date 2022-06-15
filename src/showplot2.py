
import argparse
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable

import pnets as pn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default='./model-semantickitti.pth', help='model path')
parser.add_argument('--idx', type=int, default=0, help='model index')
parser.add_argument('dataset', choices=['SemanticKITTI', 'ICCV17ShapeNetSeg', 'EricyiShapeNetSeg', 'Stanford3d'])
parser.add_argument('--class_choice', type=str, default='', help='class choice')
parser.add_argument('--npoints', default=2500, type=int)
parser.add_argument('--datadir', default='/data')
parser.add_argument('--cache', action='store_true')

args = parser.parse_args()
print(args)

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
print("model %d/%d" % (args.idx, len(ts)))
#ts = DataLoader(ts[args.idx], 32, True, num_workers=4, pin_memory=True)
#print(ts)
point, seg = ts[args.idx]
point_np=point
point=torch.from_numpy(point)
point = point.transpose(1, 0).contiguous()
point = point.view(1, point.shape[0], point.shape[1])
point=point.to(device)
print(point.shape)
save_path = args.model
model = pn.pointnet.PointNetSeg(K).to(device)
model=torch.load(save_path)
model.eval() 

cmap = plt.cm.get_cmap("hsv", 10)
cmap = np.array([cmap(i) for i in range(10)])[:, :3]
#gt = cmap[seg - 1, :]

pred, _, _ = model(point)
pred_choice = pred.data.max(2)[1]
print(pred_choice)

#print(pred_choice.size())
pred_color = cmap[pred_choice.numpy()[0], :]

#print(pred_color.shape)
pn.plot.plot3d(point_np, pred_color)
pn.plot.show()
