import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils import Multi_confusion, confusion_Matrix

def test(cfg):
    device = cfg.get('device')

    transform = transforms.Compose([
            transforms.Resize((224, 224)),          # 개와 고양이 사진 파일의 크기가 다르므로, Resize로 맞춰줍니다.
            # transforms.RandomHorizontalFlip(0.5),   # 50% 확률로 Horizontal Flip
            transforms.ToTensor(), 
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), # 이미지 정규화
        ])

    batch_size = 64
    criterion = criterion = torch.nn.CrossEntropyLoss().to(device)

    tst_ds = torchvision.datasets.ImageFolder(root='/home/KDT-admin/work/data6_last_1000_300/train', transform=transform)
    tst_dl = DataLoader(tst_ds, batch_size=batch_size, shuffle=True)


    model = torch.load('/home/KDT-admin/work/kbc/tst_model/resnet101_final_crop_37.88.pth')
    model = model.to(device)


    pred, true_label, tst_losses, tst_acces = Multi_confusion(model, tst_dl, device, criterion)

    pred_flat = [item for sublist in pred for item in sublist]
    true_label_flat = [item for sublist in true_label for item in sublist]

    confusion_Matrix(pred_flat, true_label_flat)

def get_args_parser(add_help=True):
  import argparse
  
  parser = argparse.ArgumentParser(description="?", add_help=add_help)
  parser.add_argument("-c", "--config", default="./config.py", type=str, help="configuration file")
  parser.add_argument("-m", "--model", default="?", type=str, help="configuration file")

  return parser

if __name__ == "__test__":
  args = get_args_parser().parse_args()
  exec(open(args.config).read())
  test(config)
