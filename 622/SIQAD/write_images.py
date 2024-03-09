import os
import random
import csv
import pandas as pd


def get_images(filename):
    # 读取images.csv文件中的每一行数据,并存入相应的训练集、验证集和测试集
    total_images = []
    read_path = os.path.join(os.path.dirname(__file__), filename)
    with open(read_path, 'r', encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        rows = [row for row in reader]
    for i in range(len(rows)):
        total_images.append([x for x in rows[i] if x != ''])
    train_images = total_images[0]
    val_images = total_images[1]
    test_images = total_images[2]
    train_images = [x.split("_")[0] for x in train_images]
    val_images = [x.split("_")[0] for x in val_images]
    test_images = [x.split("_")[0] for x in test_images]
    train_images = list(set(train_images))
    val_images = list(set(val_images))
    test_images = list(set(test_images))
    return train_images, val_images, test_images


dataset_root = '/data2/cyk/dataset/SIQAD/'
dataset_reference_dir = os.path.join(dataset_root, 'references')
scores_csv = os.path.join(dataset_root, 'image_labeled_by_score.csv')
csv_data = pd.read_csv(scores_csv, sep=',', usecols=['image', 'dmos', 'style'])
image_score_dict = {}
image_style_dict = {}
for i in range(len(csv_data)):
    image_score_dict[csv_data.iloc[i].image] = csv_data.iloc[i].dmos  # images->dmos
    image_style_dict[csv_data.iloc[i].image] = csv_data.iloc[i].style  # images->style
images = list(image_score_dict)

train_tag = ['cim1', 'cim3', 'cim4', 'cim5', 'cim7', 'cim8', 'cim12', 'cim13', 'cim14', 'cim15', 'cim18', 'cim19']
val_tag = ['cim20', 'cim6', 'cim9', 'cim10']
test_tag = ['cim11', 'cim16', 'cim17', 'cim2']
# ['cim12', 'cim15', 'cim16', 'cim20'] 12:0.9463 15:0.8805 16:0.9312 20:0.9024 2:0.9133 7:0.9289 13:0.9214
train_images = []
val_images = []
test_images = []
for image in images:
    image_tag = image.split("_")[0]  # 获取数据集中每张图像对应的tag
    # 对数据集中的每幅图像按标签划分到训练集，验证集和测试集
    if image_tag in train_tag:
        train_images.append(image)
    if image_tag in val_tag:
        val_images.append(image)
    if image_tag in test_tag:
        test_images.append(image)

print(len(train_images))
print(len(val_images))
print(len(test_images))
# 将训练集、验证集和测试集中的图片名按行存入images.csv文件
images_list = []
images_list.append(train_images)
images_list.append(val_images)
images_list.append(test_images)
write = pd.DataFrame(data=images_list)
write_path = os.path.join(os.path.dirname(__file__), 'image_p1.csv')
print(write_path)
assert os.path.isfile(write_path), f'File not found, {dataset_root}'
write.to_csv(write_path, index=False, header=False)
