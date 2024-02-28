import torch
from model import *

config = {
    'files':{
        'trn_root': '/home/KDT-admin/work/data6_last_1000_300/train',
        'val_root': '/home/KDT-admin/work/data6_last_1000_300/validation',
        'tst_root': '/home/KDT-admin/work/data6_last_1000_300/test'
    },
    
    'model': resnet101(),
    'device': torch.device("cuda:3" if torch.cuda.is_available() else "cpu"),
    
    'trn_params': {
        'lr': 0.001,
        'epoch': 2,
        'batch_size': 64,
        'criterion': torch.nn.CrossEntropyLoss()
    }
}