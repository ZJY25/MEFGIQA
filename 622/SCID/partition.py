import os
import random
import csv
import pandas as pd

dataset_root = '/data2/cyk/dataset/SCID/'
dataset_reference_dir = os.path.join(dataset_root, 'ReferenceSCIs')
scores_csv = os.path.join(dataset_root, 'image_labeled_by_score_SCID.csv')
csv_data = pd.read_csv(scores_csv, sep=',', usecols=['image', 'mos', 'style'])
print(f'csv_data length: {len(csv_data)}')

files = os.listdir(dataset_reference_dir)
tag = []
for file in files:
    tag.append(file.split('.')[0])

train_tag = random.sample(tag, int(len(tag) * 0.6))
tag = list(set(tag) - set(train_tag))
val_tag = random.sample(tag, int(len(tag) * 0.5))
tag = list(set(tag) - set(val_tag))
test_tag = random.sample(tag, int(len(tag)))

train_tag.sort()
val_tag.sort()
test_tag.sort()
print(train_tag)
print(val_tag)
print(test_tag)
image_score_dict = {}
image_style_dict = {}

for i in range(len(csv_data)):
    image_score_dict[csv_data.iloc[i].image] = csv_data.iloc[i].mos  # images->dmos
    image_style_dict[csv_data.iloc[i].image] = csv_data.iloc[i].style  # images->style

images = list(image_score_dict)

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

train_images.sort()
val_images.sort()
test_images.sort()
print(len(train_images))
print(len(val_images))
print(len(test_images))

# 将训练集、验证集和测试集中的图片名按行存入images.csv文件
images_list = []
images_list.append(train_images)
images_list.append(val_images)
images_list.append(test_images)
# print(len(images_list))
# name=['train_images_1','train_images_2','val_images_1','val_images_2','test_images']
write = pd.DataFrame(data=images_list)
write_path = os.path.join(os.path.dirname(__file__), 'image24.csv')
print(write_path)
assert os.path.isfile(write_path), f'File not found, {dataset_root}'
write.to_csv(write_path, index=False, header=False)