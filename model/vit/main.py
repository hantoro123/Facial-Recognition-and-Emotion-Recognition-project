from dataset import CustomDataset
from model import *
from train import *
from plot import *
from test import *
from tqdm.auto import trange
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time


def main(cfg, args):
  start_time = time.time()

  # parameters
  train_params = cfg.get('train_params')
  model_params = cfg.get('model').get('model_params')
  loader_params = train_params.get('data_loader_params')

  # select dataset
  crop_data = cfg.get('files').get('crop')
  origin_data = cfg.get('files').get('origin')

  if cfg.get('model').get('data') == 'crop':
    trn_img_path = crop_data.get('trn_img_path')
    val_img_path = crop_data.get('val_img_path')
    tst_img_path = crop_data.get('tst_img_path')
    print('Dataset : Crop data')

  else:
    trn_img_path = origin_data.get('trn_img_path')
    val_img_path = origin_data.get('val_img_path')
    tst_img_path = origin_data.get('tst_img_path')
    print('Dataset : Original data')

  # Composing dataset 
  train_dataset, val_dataset, tst_dataset = CustomDataset(trn_img_path, val_img_path, tst_img_path, loader_params.get('resize')).create_dataset()
  
  train_loader = DataLoader(train_dataset, batch_size=model_params.get('batch_size'), shuffle=True)
  val_loader = DataLoader(val_dataset, batch_size=model_params.get('batch_size'), shuffle=False)
  test_loader = DataLoader(tst_dataset, batch_size=model_params.get('batch_size'), shuffle=False)
  
  print('Dataset setting complete!')
  
  # CUDA device setting
  device = train_params.get('device')
  print(f'{device} setting is completed!')

  # model
  model = cfg.get('model').get('model_name')
  model.to(device)

  # loss 
  criterion = train_params.get('criterion')

  # optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=train_params.get('lr'))

  # model train
  model, trn_losses, trn_acces, val_losses, val_acces = train_model(
                                cfg=cfg, 
                                model=model, 
                                criterion=criterion, 
                                optimizer=optimizer, 
                                epochs=train_params.get('epochs'), 
                                train_loader=train_loader, 
                                val_loader=val_loader,
                                device=device)
  
  print('Training completed!')

  # Generate plots  
  one_item_plot(trn_losses, val_losses, 'loss', train_params.get('lr'), model_params.get('batch_size'), train_params.get('epochs'))
  one_item_plot(trn_acces, val_acces, 'acc', train_params.get('lr'), model_params.get('batch_size'), train_params.get('epochs'))

  print('2 plot results saved')

  # Load trained weight model
  model.load_state_dict(torch.load(cfg.get('files').get('model_pth')))
  # model.load_state_dict(torch.load('/home/KDT-admin/work/bsb/ViT/best/re_tViT_(epoch:30_lr:0.0001_batch:64)_crop.pth'))
  # model.to(device)

  # Test
  pred, true_label = test_model(cfg=cfg,
                                model=model, 
                                criterion=criterion, 
                                tst_loader=test_loader, 
                                device=device)

  # Generate confusion matrix
  c_matrix(pred, true_label)
  print('Confusion matrix creation completed')

def get_args_parser(add_help=True):
  import argparse
  
  parser = argparse.ArgumentParser(description="ViT model process", add_help=add_help)
  parser.add_argument("-c", "--config", default="./config.py", type=str, help="configuration file")
  parser.add_argument("-e", "--eval", default=False, type=bool, help="are you going to do the evaluation? True/False")

  return parser


if __name__ == "__main__":
  args = get_args_parser().parse_args()
  exec(open(args.config).read())
  main(config, args)