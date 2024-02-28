from torchvision import transforms, datasets, models
import torchvision
from torch.utils.data import Dataset, DataLoader

def get_dataloader(img_path,  batch_size):
  transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
  ])

  train_img_path = img_path + '/train'         # '/home/KDT-admin/work/data3/train'
  train_dataset = torchvision.datasets.ImageFolder(root=train_img_path, transform=transform)
  train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
  val_img_path = img_path + '/validation'         # '/home/KDT-admin/work/data3/train'
  val_dataset = torchvision.datasets.ImageFolder(root=val_img_path, transform=transform)
  val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
  return train_data_loader, val_data_loader