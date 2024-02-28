import torch
import torchvision
from torchvision import transforms

class CustomDataset():
  def __init__(self,img_train_path, img_val_path, resize):
    self.img_train_path = img_train_path
    self.img_val_path = img_val_path
    self.resize = resize
  
  def create_dataset(self):
    transform = transforms.Compose([
    transforms.Resize((self.resize,self.resize)),
    transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.ImageFolder(root=self.img_train_path, transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(root=self.img_val_path, transform=transform)

    return train_dataset, val_dataset