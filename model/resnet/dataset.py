import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

train_transform = transforms.Compose([
        transforms.Resize((224, 224)),          
        transforms.ToTensor(), 
    ])

test_transform = transforms.Compose([
        transforms.Resize((224, 224)),      
        transforms.ToTensor(), 
    ])

def LoadData(trn_root, tst_root, batch_size):
    train_ds = torchvision.datasets.ImageFolder(root=trn_root, transform=train_transform)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_ds = torchvision.datasets.ImageFolder(root=tst_root,transform=test_transform)
    val_dl = DataLoader(val_ds,batch_size=batch_size, shuffle=False)

    return train_dl, val_dl

def LoadTestData(tst_root, batch_size):
    tst_ds = torchvision.datasets.ImageFolder(root=tst_root, transform=test_transform)
    tst_dl = DataLoader(tst_ds, batch_size=batch_size, shuffle=True)

    return tst_dl