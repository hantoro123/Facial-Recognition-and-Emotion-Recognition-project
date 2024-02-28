import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *

config = {
    'files':{
        'img_train_path': '/home/KDT-admin/work/data6_last_1000_300/train',
        'img_val_path':'/home/KDT-admin/work/data6_last_1000_300/validation',
    },

    'resnext' : {
        'model' : resnext101(),
        'model_params' : {
            'batch_size':256,
            'num_class':7,
            'num_epochs':2,
            'num_workers':0,
            'use_gpu':True,
            'gpus':'0',  # cuda:1로 수정
            'print_freq':10,
            'save_epoch_freq':1,
            'save_path':"output",
            'resume':"",
            'start_epoch':0,
            'data_size':224,
            'activation' : F.relu,
        },
        "pretrained":'resnext101_32x8d',
    },

    'train_params': {
        'data_loader_params' : {
            'resize': 224,
            'suffle' : True,
            'tst_size' : 20,
        },
        'loss' : F.mse_loss,
        'optim' : torch.optim.AdamW,
        'optim_params' : {
            'lr' : 0.0001,
            'name':'Adam',
            'weight_decay':0.0001,
            'step_size':30,
            'gamma':0.1,
        },
        'metric' : '',
        'device' : torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'epochs' : 2, 
        'criterion':nn.CrossEntropyLoss(),
  }
}