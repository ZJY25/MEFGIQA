# !/usr/bin/env python
# coding: utf-8
import numpy as np
import time
import pandas as pd
import random
import torch
import os
import csv
import logging
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = '3'
# 设置随机数种子
seed = 0
sys.path.extend(['/data2/cyk/MyIQA2New', '/data2/cyk/MyIQA2New'])


# nohup python3 -u trainSIQAD.py > result.log 2>&1 &


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


setup_seed(seed)


import torch
import torch.optim.lr_scheduler as lr_scheduler
from math import sqrt
from models1.model_base import MyNet
from utils.SIQAD1.data import train_loader, val_loader, test_loader
from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn.metrics import mean_squared_error
from tensorboardX import SummaryWriter

device_ids = [0]  # 指定所使用的的GPU设备
device = "cuda:0"

case_dir = os.path.join('.')
if not os.path.exists(case_dir):
    os.makedirs(case_dir)
fmt = logging.Formatter('%(levelname)s %(message)s')
logger = logging.getLogger()
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(fmt)
file_handler = logging.FileHandler(os.path.join(case_dir, 'Case_SIQAD.log'))
file_handler.setFormatter(fmt)
logger.setLevel(logging.INFO)
stream_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.INFO)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

logger.info(f'random seed: {seed}')
writer = SummaryWriter('./log')
model_pt_dir = os.path.join(os.path.dirname(__file__), 'pt_dir')
assert os.path.isdir(model_pt_dir), f'File not found, {os.path.dirname(__file__)}'

test_images = []
read_path = os.path.join(os.path.join(os.path.join(os.path.abspath(os.path.join(os.getcwd(), "..")), 'utils'), 'SIQAD1'), 'test.csv')
print(read_path)
with open(read_path, 'r', encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
for i in range(len(rows)):
    test_images.append([x for x in rows[i] if x != ''])

# Framework
print("Start build model")
net = MyNet()
net = net.to(device)
print("End build model")

criterion = torch.nn.MSELoss()
resnet_params = list(map(id, net.resnet.parameters()))
base_params = filter(lambda p: id(p) not in resnet_params, net.parameters())
optimizer = torch.optim.Adam([
    {'params': base_params},
    {'params': net.resnet.parameters(), 'lr': 5e-3}], lr=1e-2
)
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 40, 60, 80, 100], gamma=0.1)
print('------------------train------------------')
val_min_loss = 1e10
best_srocc = 0.0

for epoch in range(1, 121):
    if epoch in range(1, 21):
        net.freezing()
    else:
        net.unfreezing()
    # train
    start_time = time.time()
    net.train()
    train_loss = 0.0
    optimizer.zero_grad()

    for batch_idx, data in enumerate(train_loader):
        images, log_images, scores, style, grade = data['image'].to(device), data['log_image'].to(device), data['score'].to(device), data['style'].to(device), data['grade'].to(device)
        scores = scores.float()
        pred_score = net(images, log_images)
        loss = criterion(pred_score, scores)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()

    if not torch.isfinite(loss):
        print('WARNING: non-finite loss, ending training ', loss)
        sys.exit(1)

    end_time = time.time()
    lr_1 = optimizer.state_dict()['param_groups'][0]['lr']
    lr_2 = optimizer.state_dict()['param_groups'][1]['lr']
    logger.info("train-epoches: %d, time: %.4f, loss: %.4f, learning_rate: %f" % (
        epoch, end_time - start_time, train_loss / len(train_loader.dataset), lr_1))
    writer.add_scalar('train_loss', train_loss / len(train_loader.dataset), global_step=epoch)
    scheduler.step()

    # val
    net.eval()
    val_loss = 0.0
    with torch.no_grad():
        predsc_1 = []
        gt_score_1 = []

        for batch_idx, data in enumerate(val_loader):
            images, log_images, scores, style, grade = data['image'], data['log_image'], data['score'].to(device), data['style'].to(device), data['grade'].to(device)
            scores = scores.float()
            sum_score = 0.0
            sum_loss = 0.0
            images = torch.stack(images, dim=1)
            images = images.squeeze()
            log_images = torch.stack(log_images, dim=1)
            log_images = log_images.squeeze(dim=0)
            for i in range(len(images)):
                image = images[i:i + 1, :, :, :]
                image = image.to(device)
                log_image = log_images[i:i + 1, :, :, :]
                log_image = log_image.to(device)
                score = net(image, log_image)
                loss = criterion(score, scores)
                sum_score += torch.sum(score).item()
                sum_loss += loss

            predsc_1.append(sum_score / len(images))
            gt_score_1.append(scores.item())
            val_loss += sum_loss / len(images)
            with open('./model_result/SIQAD/val/result_{}.txt'.format(epoch), 'a') as file_handle:
                file_handle.write(
                    "{}\n".format(str(predsc_1[len(predsc_1) - 1]) + " " + str(gt_score_1[len(gt_score_1) - 1])))

        file_handle.close()
        ratings_i = np.vstack(gt_score_1)
        predictions_i_sc = np.vstack(predsc_1)

        a = ratings_i[:, 0]
        b = predictions_i_sc[:, 0]
        sp_1 = spearmanr(a, b)[0]
        rmse_1 = sqrt(mean_squared_error(a, b))
        pl_1 = pearsonr(a, b)[0]
        ke_1 = kendalltau(a, b)[0]

    if (val_loss / len(val_loader.dataset)) < val_min_loss:
        val_min_loss = val_loss / len(val_loader.dataset)

    lr_1 = optimizer.state_dict()['param_groups'][0]['lr']
    lr_2 = optimizer.state_dict()['param_groups'][1]['lr']
    logger.info("val-epoches: %d, val_min_loss: %.4f, val_loss: %.4f, best_srocc: %.4f" % (
        epoch, val_min_loss, val_loss / len(val_loader.dataset), best_srocc))
    logger.info("val-epoches: %d, PLCC: %.4f, SROCC: %.4f, KROCC: %.4f, RMSE: %.4f" % (epoch, pl_1, sp_1, ke_1, rmse_1))
    writer.add_scalar('val_loss', val_loss / len(val_loader.dataset), global_step=epoch)
    writer.add_scalar('val_PLCC', pl_1, global_step=epoch)
    writer.add_scalar('val_SROCC', sp_1, global_step=epoch)
    writer.add_scalar('val_RMSE', rmse_1, global_step=epoch)

    # test
    result_data = []
    col_result = ['epoch', 'test_images', 'pred_score', 'gt_score']
    score_data = []
    col_score = ['epoch', 'SROCC', 'PLCC', 'KROCC', 'RMSE']
    with open('./model_result/SIQAD/test/result_{}.txt'.format(epoch), 'a') as file_handle:  # .txt可以不自己新建,代码会自动新建
        file_handle.write("{}\n".format("predscore" + " " + "GT"))
    net.eval()
    test_loss = 0.0
    with torch.no_grad():
        predsc_1 = []
        gt_score_1 = []

        for batch_idx, data in enumerate(test_loader):
            images, log_images, scores, style, grade = data['image'], data['log_image'], data['score'].to(device), data[
                'style'].to(device), data['grade'].to(device)
            scores = scores.float()
            sum_score = 0.0
            sum_loss = 0.0
            images = torch.stack(images, dim=1)
            images = images.squeeze()
            log_images = torch.stack(log_images, dim=1)
            log_images = log_images.squeeze(dim=0)
            for i in range(len(images)):
                image = images[i:i + 1, :, :, :]
                image = image.to(device)
                log_image = log_images[i:i + 1, :, :, :]
                log_image = log_image.to(device)
                score = net(image, log_image)
                loss = criterion(score, scores)
                sum_score += torch.sum(score).item()
                sum_loss += loss

            predsc_1.append(sum_score / len(images))
            gt_score_1.append(scores.item())
            result_data.append(
                [epoch, os.path.basename(test_images[batch_idx][0].split('/')[-1]), predsc_1[len(predsc_1) - 1],
                 gt_score_1[len(gt_score_1) - 1]])
            test_loss += sum_loss / len(images)
            with open('./model_result/SIQAD/test/result_{}.txt'.format(epoch), 'a') as file_handle:
                file_handle.write(
                    "{}\n".format(str(predsc_1[len(predsc_1) - 1]) + " " + str(gt_score_1[len(gt_score_1) - 1])))

        file_handle.close()
        write = pd.DataFrame(columns=col_result, data=result_data)
        write_path = os.path.join(os.path.dirname(__file__), 'SIQAD_TestResult.csv')
        assert os.path.isfile(write_path), f'File not found, {write_path}'
        write.to_csv(write_path, mode='a', header=True, index=False)

        ratings_i = np.vstack(gt_score_1)
        predictions_i_sc = np.vstack(predsc_1)
        a = ratings_i[:, 0]
        b = predictions_i_sc[:, 0]
        sp_1 = spearmanr(a, b)[0]
        rmse_1 = sqrt(mean_squared_error(a, b))
        pl_1 = pearsonr(a, b)[0]
        ke_1 = kendalltau(a, b)[0]
        score_data.append([epoch, sp_1, pl_1, ke_1, rmse_1])
        write = pd.DataFrame(columns=col_score, data=score_data)
        write_path = os.path.join(os.path.dirname(__file__), 'SIQAD_ScoreData.csv')
        assert os.path.isfile(write_path), f'File not found, {write_path}'
        write.to_csv(write_path, mode='a', header=True, index=False)
        logger.info(
            "test-epoches: %d, PLCC: %.4f, SROCC: %.4f, KROCC: %.4f, RMSE: %.4f" % (epoch, pl_1, sp_1, ke_1, rmse_1))
        writer.add_scalar('test_PLCC', pl_1, global_step=epoch)
        writer.add_scalar('test_SROCC', sp_1, global_step=epoch)
        writer.add_scalar('test_RMSE', rmse_1, global_step=epoch)

        if sp_1 > best_srocc:
            best_srocc = sp_1
            torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, os.path.join(model_pt_dir, 'best-checkpoint.pt'))
            logger.info(f"The best srocc is: %.4f" % best_srocc)