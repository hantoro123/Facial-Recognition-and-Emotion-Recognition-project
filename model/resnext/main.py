from customdataset import CustomDataset
from model import *
from train import *
from plot import *
from tqdm.auto import trange
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time


def main(cfg, args):
  start_time = time.time() 

  # 각 파라미터
  train_params = cfg.get('train_params')
  model_params = cfg.get('resnext').get('model_params')
  optim_params = train_params.get('optim_params')
  data_loader_params = train_params.get('data_loader_params')
  print("parameter 완료")

  train_dataset, val_dataset = CustomDataset(img_train_path=cfg.get('files').get('img_train_path'), img_val_path=cfg.get('files').get('img_val_path'), resize=data_loader_params.get('resize')).create_dataset()

  dataloaders = {
        'train': DataLoader(train_dataset, batch_size=model_params.get('batch_size'), shuffle=True, num_workers=model_params.get('num_workers')),
        'val': DataLoader(val_dataset, batch_size=model_params.get('batch_size'), shuffle=False, num_workers=model_params.get('num_workers'))
    }
  dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
  print("data set 완료")

  # CUDA 장치 설정
  device = train_params.get('device')
  print(f"{device} 완료")
  # pretrained 여부에 따른 모델 생성 및 초기화
  if args.pretrained:
    pretrained_model = torch.hub.load('pytorch/vision:v0.10.0', cfg.get('resnext').get('pretrained'), pretrained=args.pretrained)
    num_ftrs = pretrained_model.fc.in_features
    pretrained_model.fc = nn.Linear(num_ftrs, model_params.get('num_class'))
    model = pretrained_model.to(device)

    # 모델의 레이어 중 일부를 고정합니다. 여기서는 마지막 Fully Connected Layer를 제외한 모든 레이어를 고정합니다.
    for param in pretrained_model.parameters():
        param.requires_grad = False

    # 마지막 Fully Connected Layer의 파라미터들만 학습할 수 있도록 설정합니다.
    for param in pretrained_model.fc.parameters():
        param.requires_grad = True
  else:
    # model = cfg.get('resnext')
    model = resnext101(num_classes=model_params.get('num_class')).to(device)
  
  print(f"{model} 완료")
  resume = model_params.get('resume')
  if resume:
    if os.path.isfile(resume):
        print(("=> loading checkpoint '{}'".format(resume)))
        checkpoint = torch.load(resume)
        base_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.state_dict().items())}
        model.load_state_dict(base_dict)
    else:
        print(("=> no checkpoint found at '{}'".format(resume)))

  print("check point 완료")
  # 손실 함수 정의
  criterion = train_params.get('criterion')

  # 옵티마이저 및 스케줄러 설정
  optimizer_ft = optim.Adam(model.parameters(), lr=optim_params.get('lr'), weight_decay=optim_params.get('weight_decay'))
  exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=optim_params.get('step_size'), gamma=optim_params.get('gamma'))

  print("손실함수 및 옵티마이져 등 완료")
  # 모델 훈련
  model, trn_losses, trn_acces, val_losses, val_acces = train_model(cfg=cfg,
                      model=model,
                      criterion=criterion,
                      optimizer=optimizer_ft,
                      scheduler=exp_lr_scheduler,
                      num_epochs=model_params.get('num_epochs'),
                      dataloaders=dataloaders,
                      dataset_sizes=dataset_sizes)

  print("학습 완료")
  one_item_plot(trn_losses,val_losses, 'losses', optim_params.get('lr'), model_params.get('batch_size'), model_params.get('num_epochs'))
  one_item_plot(trn_acces, val_acces, 'acc', optim_params.get('lr'), model_params.get('batch_size'), model_params.get('num_epochs'))

  save_loss_acc(cfg,trn_losses, val_losses, trn_acces, val_acces)
  print("plot및 저장 완료")

  model_save(model,cfg,start_time)

def get_args_parser(add_help=True):
  import argparse
  
  parser = argparse.ArgumentParser(description="Pytorch K-fold Cross Validation", add_help=add_help)
  parser.add_argument("-c", "--config", default="./config.py", type=str, help="configuration file")
  parser.add_argument("-p", "--pretrained", default=True, type=bool, help="how to pretraied weight True/False")
  parser.add_argument("-e", "--eval", default=False, type=bool, help="are you going to do the evaluation? True/False")

  return parser

if __name__ == "__main__":
  args = get_args_parser().parse_args()
  exec(open(args.config).read())
  main(config, args)