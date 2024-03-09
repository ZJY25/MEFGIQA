from PIL import Image
import torch
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
from CAM.model import MyNet


def image_process1(image_path):
    # 读取数据集中的图片
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    # 图片预处理
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize
    ])
    img_pil = Image.open(image_path)
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    return img_variable


def image_process2(image_path):
    # 读取数据集中的图片
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    # 图片预处理
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # normalize
    ])
    img_pil = Image.open(image_path)
    img_tensor = preprocess(img_pil)
    img_variable = Variable(img_tensor.unsqueeze(0))
    return img_variable


# 定义计算CAM的函数
def returnCAM(feature_conv, weight_softmax):
    # 类激活图上采样到 256 x 256
    size_upsample = (256, 256)
    # print(feature_conv.shape)
    # print(weight_softmax.shape)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    # 将权重赋给卷积层：这里的weigh_softmax.shape为(1000, 512)
    # 				feature_conv.shape为(1, 512, 13, 13)
    # weight_softmax[class_idx]由于只选择了一个类别的权重，所以为(1, 512)
    # feature_conv.reshape((nc, h * w))后feature_conv.shape为(512, 169)
    cam = weight_softmax.dot(feature_conv.reshape((nc, h * w)))
    print(cam.shape)  # 矩阵乘法之后，为各个特征通道赋值。输出shape为（1，169）
    cam = cam.reshape(h, w)  # 得到单张特征图
    # 特征图上所有元素归一化到 0-1
    cam_img = (cam - cam.min()) / (cam.max() - cam.min())
    # 再将元素更改到　0-255
    cam_img = np.uint8(255 * cam_img)
    output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam


def hook_feature(module, input, output):
    # print(output)
    features_blobs.append(output.data.cpu().numpy())



if __name__ == "__main__":
    # 加载预训练模型
    net = MyNet()
    net.load_state_dict(torch.load('/data2/cyk/MyIQA2New/CAM/best-checkpoint.pt')['model_state_dict'])
    net.eval()
    # 加载图片
    dis_image = image_process1('./cim1_1_1_D.bmp')
    edge_image = image_process2('./cim1_1_1_Edge.bmp')
    # 获取特征图
    features_blobs = []  # 后面用于存放特征图
    # 获取 features 模块的输出
    module_name = 'efm1'
    net._modules.get(module_name).register_forward_hook(hook_feature)
    params = {}
    for name, param in net.named_parameters():
        print(name)
        params[name] = param
    layer_name = 'efm1.conv2d.block.1.weight'
    print((params[layer_name].data.numpy()).shape)
    weight_softmax = np.mean(np.squeeze(params[layer_name].data.numpy()), axis=0)
    # weight_softmax = np.mean(weight_softmax, axis=1)
    logit = net(dis_image, edge_image)  # 计算输入图片通过网络后的输出值
    # 对概率最高的类别产生类激活图
    print(features_blobs[0].shape)
    CAMs = returnCAM(features_blobs[0], weight_softmax)
    # 融合类激活图和原始图片
    img = cv2.imread('./cim1_1_1_D.bmp')
    height, width, _ = img.shape
    heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (width, height)), cv2.COLORMAP_JET)
    result = heatmap * 0.3 + img * 0.7
    cv2.imwrite('CAM.jpg', result)