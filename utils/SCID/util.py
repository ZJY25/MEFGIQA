import numbers
import torch
import csv
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from PIL import Image


def save_image_tensor2pillow(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为pillow
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为numpy
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # 转成pillow
    im = Image.fromarray(input_tensor)
    im.save(filename)


def Image2Patch(image, size, stride):
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    elif isinstance(size, (tuple, list)) and len(size) == 1:
        size = (size[0], size[0])

    if len(size) != 2:
        raise ValueError("Please provide only two dimensions (h, w) for size.")

    H, W = image.size

    if size[0] > H or size[1] > W:
        raise ValueError(f'output_size > image size, output_size:{size}, image_size:{image.size}')

    patches = []

    for i in range(0, H, stride):

        if i + size[0] >= H:
            i = H - size[0] - 1

        for j in range(0, W, stride):
            if j + size[1] >= W:
                j = W - size[1] - 1

            patch = F.crop(image, j, i, size[0], size[1])
            patches.append(patch)

            if j == W - size[1] - 1:
                break

        if i == H - size[0] - 1:
            break

    return patches


class Train_IQADataset(Dataset):
    def __init__(self, csv_path, transform=None):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            rows = [row for row in reader]

        images = []
        for i in range(0, len(rows)):
            images.append((rows[i][0], rows[i][1], float(rows[i][2]), int(rows[i][3]), int(rows[i][4])))   # 分别表示训练集失真图片路径、对应的log特征图路径、 质量分数、失真类型、失真等级
        self.images = images
        self.transform = transform

    def __getitem__(self, index):
        img_path, LogImg_path, score, style, grade = self.images[index]
        image = Image.open(img_path)
        Log_image = Image.open(LogImg_path)
        sample = {'image': image, 'log_image': Log_image, 'score': score, 'style': style, 'grade': grade}
        seed = np.random.randint(2147483647)  # make a seed with numpy generator

        if self.transform is not None:
            torch.manual_seed(seed)  # apply this seed to img tranfsorms
            sample['image'] = self.transform(sample['image'])
            torch.manual_seed(seed)  # apply this seed to log_img tranfsorms
            sample['log_image'] = self.transform(sample['log_image'])
        return sample

    def __len__(self):
        return len(self.images)


class Test_IQADataset(Dataset):
    def __init__(self, csv_path, transform=None):
        with open(csv_path, 'r') as f:
            reader = csv.reader(f)
            rows = [row for row in reader]
        images = []
        for i in range(0, len(rows)):
            images.append((rows[i][0], rows[i][1], float(rows[i][2]), int(rows[i][3]), int(rows[i][4])))   # 分别表示训练集失真图片路径、对应的log特征图路径、 质量分数、失真类型、失真等级
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path, LogImg_path, score, style, grade = self.images[index]
        image = Image.open(img_path)
        Log_image = Image.open(LogImg_path)
        image_list = Image2Patch(image, size=(224, 224), stride=112)
        Log_image_list = Image2Patch(Log_image, size=(224, 224), stride=112)
        sample = {'image': [], 'log_image': [], 'score': score, 'style': style, 'grade': grade}
        if self.transform:
            for key, value in enumerate(image_list):
                sample['image'].append(self.transform(value))
            for key, value in enumerate(Log_image_list):
                sample['log_image'].append(self.transform(value))
        return sample