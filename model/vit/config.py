import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *



config = {
  'files':{
    'model_pth' : '/home/KDT-admin/work/bsb/ViT/best/re_tViT_(epoch:30_lr:0.0001_batch:64)_crop.pth',
    'origin':{'trn_img_path' : '/home/KDT-admin/work/data6_last_1000_300/train',
              'val_img_path' : '/home/KDT-admin/work/data6_last_1000_300/validation',
              'tst_img_path' : '/home/KDT-admin/work/data6_last_1000_300/test'
              }, 
    'crop': {'trn_img_path' : '/home/KDT-admin/work/data6_last_crop/train',
            'val_img_path' : '/home/KDT-admin/work/data6_last_crop/validation',
            'tst_img_path' : '/home/KDT-admin/work/data6_last_crop/test'
    }
    
  },

  'model' :{
    'data' : 'crop',
    'model_name' : ViT(),
    'model_params' : {
      'batch_size' : 64,
      'imgsz' : 224,
      'patch_size' : 16, # imgsz 약수
      'n_classes' : 7,
    }
  },
  
  'train_params' : {
    'data_loader_params' : {
      'resize' : 224,
    },
    # 'optim' : torch.optim.Adam(model.parameters(), lr=lr),
    'lr' : 0.0001,
    'epochs' : 5,
    'criterion' : nn.CrossEntropyLoss(),
    'device' : torch.device('cuda:2' if torch.cuda.is_available() else 'cpu'),

  }
}