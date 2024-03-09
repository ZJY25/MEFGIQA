import os
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
from utils.SCID.util import *

train_batch_size = 30
val_batch_size = 1
test_batch_size = 1

# 对训练集进行数据预处理，并构造加载器
train_csvpath = os.path.join(os.path.dirname(__file__), 'train.csv')
train_data_transforms = transforms.Compose([
    transforms.RandomCrop((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])
train_dataset = Train_IQADataset(train_csvpath, transform=train_data_transforms)
train_loader = DataLoader(train_dataset, batch_size=train_batch_size, drop_last=True, shuffle=True, num_workers=0)

# 对验证集进行数据预处理，并构造加载器
val_csvpath = os.path.join(os.path.dirname(__file__), 'val.csv')
val_data_transforms = transforms.Compose([
    transforms.ToTensor()
])
val_dataset = Test_IQADataset(val_csvpath, transform=val_data_transforms)
val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=0)

# 对测试集进行数据预处理，并构造加载器
test_csvpath = os.path.join(os.path.dirname(__file__), 'test.csv')
test_data_transforms = transforms.Compose([
    transforms.ToTensor()
])
test_dataset = Test_IQADataset(test_csvpath, transform=test_data_transforms)
test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=0)