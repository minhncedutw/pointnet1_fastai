'''
    File name: robot-grasping
    Author: minhnc
    Date created(MM/DD/YYYY): 9/1/2018
    Last modified(MM/DD/YYYY HH:MM): 9/1/2018 9:25 AM
    Python Version: 3.5
    Other modules: [tensorflow-gpu 1.3.0]

    Copyright = Copyright (C) 2017 of NGUYEN CONG MINH
    Credits = [None] # people who reported bug fixes, made suggestions, etc. but did not actually write the code
    License = None
    Version = 0.9.0.1
    Maintainer = [None]
    Email = minhnc.edu.tw@gmail.com
    Status = Prototype # "Prototype", "Development", or "Production"
    Code Style: http://web.archive.org/web/20111010053227/http://jaynes.colorado.edu/PythonGuidelines.html#module_formatting
'''

#==============================================================================
# Imported Modules
#==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
import time
import random

import torchvision.transforms as tt
from fastai.conv_learner import *

from DATA.ARLab.arlab_dataloader import PartDataset
from pointnet import PointNetDenseCls2

#==============================================================================
# Constant Definitions
#==============================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--directory', type=str, default='E:/PROJECTS/NTUT/PointNet/pointnet1_pytorch/DATA/Shapenet/shapenetcore_partanno_segmentation_benchmark_v0', help='data directory')
parser.add_argument('--num_points', type=int, default=1024, help='number of input points')
parser.add_argument('--batch_size', type=int, default=4, help='input batch size')
parser.add_argument('--num_workers', type=int, help='number of data loading workers', default=0) # Notice on Ubuntu, number worker should be 4; but on Windows, number worker HAVE TO be 0
parser.add_argument('--num_epoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--out_file', type=str, default='DATA/ARLab/seg',  help='output folder')
parser.add_argument('--model_path', type=str, default='',  help='model path')

opt = parser.parse_args()
print(opt)

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

#==============================================================================
# Function Definitions
#==============================================================================


#==============================================================================
# Main function
#==============================================================================
def main(argv=None):
    print('Hello! This is XXXXXX Program')

    trn_ds = PartDataset(root=opt.directory, npoints=opt.num_points, classification=False, class_choice=['Airplane'])
    val_ds = PartDataset(root=opt.directory, npoints=opt.num_points, classification=False, class_choice=['Airplane'], train=False)
    num_classes = trn_ds.num_seg_classes

    trn_dl = DataLoader(dataset=trn_ds, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_dl = DataLoader(dataset=val_ds, batch_size=opt.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    tes_dl = None

    model_data = ModelData(opt.directory, trn_dl, val_dl)

    model = PointNetDenseCls2(num_points=opt.num_points, k=num_classes)

    optimizer = optim.Adam
    criterion = F.cross_entropy
    learner = Learner(model_data, BasicModel(to_gpu(model)), opt_fn=optimizer, crit=criterion)

    lr = 5e-3
    learner.fit(lrs=lr, n_cycle=2, cycle_len=3)
    learner.save('gross_trained')

    lrs = np.array([lr / 100, lr / 10, lr])
    learner.fit(lrs=lrs / 10, n_cycle=2, cycle_len=10)
    learner.save('finetune_trained')

    # preds, targs = learn.TTA()
    y = learner.predict()
    x, _ = next(iter(model_data.val_dl))


if __name__ == '__main__':
    main()

'''
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))
    # dataloader = DataLoader(dataset, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))
    
    batch = next(iter(dataloader))
    print(batch)

    for i, data in enumerate(dataloader, 0):
        points, target = data
        print(points.shape, target.shape)
        break
'''
