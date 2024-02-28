import torchvision
from torchvision import transforms



class CustomDataset():
  def __init__(self, trn_img_path, val_img_path, tst_img_path, resize):
    self.trn_img_path = trn_img_path
    self.val_img_path = val_img_path
    self.tst_img_path = tst_img_path
    self.resize = resize

  def create_dataset(self):
    transform = transforms.Compose([
    transforms.Resize((self.resize, self.resize)),
    transforms.ToTensor(),
    ])

    train_dataset = torchvision.datasets.ImageFolder(root=self.trn_img_path, transform=transform)
    val_dataset = torchvision.datasets.ImageFolder(root=self.val_img_path, transform=transform)
    tst_dataset = torchvision.datasets.ImageFolder(root=self.tst_img_path, transform=transform)
    
    return train_dataset, val_dataset, tst_dataset