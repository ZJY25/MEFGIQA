import torch
import os
import csv
import pandas as pd
from math import sqrt
import numpy as np
from SIQAD.models.model2 import MyNet
from SIQAD.utils.data import test_loader
from scipy.stats import spearmanr, pearsonr, kendalltau
from sklearn.metrics import mean_squared_error

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
device_ids = [0] # 指定所使用的的GPU设备
device = "cuda:0"

model_pt_dir=os.path.join(os.path.dirname(__file__), 'pt_dir')
assert os.path.isdir(model_pt_dir), f'File not found, {os.path.dirname(__file__)}'

pt_list=os.listdir(model_pt_dir)
# pt_list.sort(key=lambda x: int(x.split('-')[0]))

test_images = []
read_path = os.path.join(os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), 'utils'), 'test.csv')
# print(read_path)
with open(read_path,'r',encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
for i in range(len(rows)):
    test_images.append([x for x in rows[i] if x != ''])

# with open('result.txt','a') as file_handle:   # .txt可以不自己新建,代码会自动新建
#     file_handle.write("{}\n".format("predscore"+" "+"GT"))

for pt in pt_list:
    model = MyNet()
    model = torch.nn.DataParallel(model, device_ids=device_ids, output_device=device_ids[0])  # 声明所有可用设备
    model = model.cuda(device=device_ids[0])
    model.load_state_dict(torch.load(os.path.join(model_pt_dir, pt))['model_state_dict'])
    model.eval()

    result_data = []
    col_result = ['model', 'test_images', 'pred_score', 'gt_score']
    score_data = []
    col_score = ['model', 'SROCC', 'PLCC', 'KROCC', 'RMSE']

    with torch.no_grad():
        predsc_1 = []
        gt_score_1 = []

        for batch_idx, data in enumerate(test_loader):
            images, log_images, scores, style, grade = data['image'], data['log_image'], data['score'].to(device), data['style'].to(device), data['grade'].to(device)
            scores = scores.float()
            sum_score = 0.0
            images = torch.stack(images, dim=1)
            images = images.squeeze()
            log_images = torch.stack(log_images, dim=1)
            log_images = log_images.squeeze(dim=0)
            for i in range(len(images)):
                image = images[i:i + 1, :, :, :]
                image = image.to(device)
                log_image = log_images[i:i + 1, :, :, :]
                log_image = log_image.to(device)
                score_1 = model(image, log_image)
                sum_score += torch.sum(score_1).item()

            predsc_1.append(sum_score / len(images))
            gt_score_1.append(scores.item())
            result_data.append([pt, os.path.basename(test_images[batch_idx][0].split('/')[-1]), predsc_1[len(predsc_1) - 1], gt_score_1[len(gt_score_1) - 1]])

            # with open('result.txt', 'a') as file_handle:  # .txt可以不自己新建,代码会自动新建
            #     file_handle.write("{}\n".format(str(predsc_1[len(predsc_1) - 1]) + " " + str(gt_score_1[len(gt_score_1) - 1])))

        # file_handle.close()
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
        score_data.append([pt, sp_1, pl_1, ke_1, rmse_1])

        print('the scores of {} are:'.format(pt))
        print('SROCC_1', sp_1)
        print('PLCC_1', pl_1)
        print('KROCC_1', ke_1)
        print('RMSE_1', rmse_1)

        write = pd.DataFrame(columns=col_score, data=score_data)
        write_path = os.path.join(os.path.dirname(__file__), 'SIQAD_ScoreData.csv')
        assert os.path.isfile(write_path), f'File not found, {write_path}'
        write.to_csv(write_path, mode='a', header=True, index=False)