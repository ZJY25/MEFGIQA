import os
import pandas as pd
import csv
from shutil import copy, rmtree


def mk_file(file_path: str):
    if os.path.exists(file_path):
        # 如果文件夹存在，则先删除原文件夹在重新创建
        rmtree(file_path)

    os.makedirs(file_path)


# Dataset prepared for training
dataset_root = '/data2/cyk/dataset/SIQAD/'
assert os.path.isdir(dataset_root), f'File not found, {dataset_root}'

dataset_DistortedImages_dir = os.path.join(dataset_root, 'DistortedImages')
assert os.path.isdir(dataset_DistortedImages_dir), f'File not found, {dataset_DistortedImages_dir}'

dataset_reference_dir = os.path.join(dataset_root, 'references')
assert os.path.isdir(dataset_reference_dir), f'File not found, {dataset_reference_dir}'

dataset_LogImages_dir = os.path.join(dataset_root, 'log_map')
assert os.path.isdir(dataset_LogImages_dir), f'File not found, {dataset_LogImages_dir}'

save_dataset_root = os.path.join(os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'DataSet'), 'SIQAD2')
assert os.path.isdir(save_dataset_root), f'File not found, {save_dataset_root}'

save_DistortedImages_dir = os.path.join(save_dataset_root, 'DistortedImages')
assert os.path.isdir(save_DistortedImages_dir), f'File not found, {save_DistortedImages_dir}'

save_LogImages_dir = os.path.join(save_dataset_root, 'LogImages')
assert os.path.isdir(save_LogImages_dir), f'File not found, {save_LogImages_dir}'

scores_csv = os.path.join(dataset_root, 'image_labeled_by_score.csv')
assert os.path.isfile(scores_csv), f'File not found. {scores_csv}'

csv_data = pd.read_csv(scores_csv, sep=',', usecols=['image', 'dmos', 'style', 'grade'])

image_score_dict = {}
image_style_dict = {}
image_grade_dict = {}

for i in range(len(csv_data)):
    image_score_dict[csv_data.iloc[i].image] = csv_data.iloc[i].dmos
    image_style_dict[csv_data.iloc[i].image] = csv_data.iloc[i].style - 1
    image_grade_dict[csv_data.iloc[i].image] = csv_data.iloc[i].grade - 1

# 从指定文件中读取划分好的训练集、验证集和测试集
train_images = []
val_images = []
test_images = []
total_images = []
read_path = os.path.join('/data2/cyk/MyIQA2New/622/SIQAD/', 'test19.csv')  # 指定划分的文件

print(read_path)
assert os.path.isfile(read_path), f'File not found, {read_path}'

with open(read_path, 'r', encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
for i in range(len(rows)):
    total_images.append([x for x in rows[i] if x != ''])

train_images = total_images[0]
val_images = total_images[1]
test_images = total_images[2]

train_images_path = []  # 存储训练集的所有图片路径
val_images_path = []  # 存储验证集的所有图片路径
test_images_path = []  # 存储测试集的所有图片路径
train_LogImages_path = []  # 存储训练集的所有图片对应的Log图路径
val_LogImages_path = []  # 存储验证集的所有图对应的Log图路径
test_LogImages_path = []  # 存储测试集的所有图片对应的Log图路径
train_images_label = []  # 存储训练集图片对应索引信息(score)
val_images_label = []  # 存储验证集图片对应索引信息(score)
test_images_label = []  # 存储测试集图片对应索引信息(score)

# 建立保存训练集的文件夹,并将训练集图片保存到训练集文件夹
col_result = ['img_path', 'LogImages_path', 'score', 'style', 'grade']
result_data = []
total_data = []
train_images_root = os.path.join(save_DistortedImages_dir, "train")
mk_file(train_images_root)
train_Logimages_root = os.path.join(save_LogImages_dir, "train")
mk_file(train_Logimages_root)

for train_image in train_images:
    # 将分配至训练集中的文件复制到相应目录
    image_path = os.path.join(dataset_DistortedImages_dir, train_image)
    train_images_path.append(image_path)
    Logimage_path = os.path.join(dataset_LogImages_dir, train_image)
    train_LogImages_path.append(Logimage_path)
    train_images_label.append(image_score_dict[train_image])

    Images_new_path = os.path.join(train_images_root, train_image)
    copy(image_path, Images_new_path)
    LogImages_new_path = os.path.join(train_Logimages_root, train_image)
    copy(Logimage_path, LogImages_new_path)
    result_data.append(
        [Images_new_path, LogImages_new_path, image_score_dict[train_image], image_style_dict[train_image],
         image_grade_dict[train_image]])
    total_data.append(
        [Images_new_path, LogImages_new_path, image_score_dict[train_image], image_style_dict[train_image],
         image_grade_dict[train_image]])
write = pd.DataFrame(columns=col_result, data=result_data)
write_path = os.path.join(os.path.dirname(__file__), 'train.csv')
assert os.path.isfile(write_path), f'File not found, {write_path}'
write.to_csv(write_path, mode='w', header=False, index=False)

# 建立保存验证集的文件夹,并将验证集图片保存到验证集文件夹
col_result = ['img_path', 'LogImages_path', 'score', 'style', 'grade']
result_data = []
val_images_root = os.path.join(save_DistortedImages_dir, "val")
mk_file(val_images_root)
val_Logimages_root = os.path.join(save_LogImages_dir, "val")
mk_file(val_Logimages_root)

for val_image in val_images:
    # 将分配至训练集中的文件复制到相应目录
    image_path = os.path.join(dataset_DistortedImages_dir, val_image)
    val_images_path.append(image_path)
    Logimage_path = os.path.join(dataset_LogImages_dir, val_image)
    val_LogImages_path.append(Logimage_path)
    val_images_label.append(image_score_dict[val_image])

    Images_new_path = os.path.join(val_images_root, val_image)
    copy(image_path, Images_new_path)
    LogImages_new_path = os.path.join(val_Logimages_root, val_image)
    copy(Logimage_path, LogImages_new_path)
    result_data.append(
        [Images_new_path, LogImages_new_path, image_score_dict[val_image], image_style_dict[val_image],
         image_grade_dict[val_image]])
    total_data.append(
        [Images_new_path, LogImages_new_path, image_score_dict[val_image], image_style_dict[val_image],
         image_grade_dict[val_image]])
write = pd.DataFrame(columns=col_result, data=result_data)
write_path = os.path.join(os.path.dirname(__file__), 'val.csv')
assert os.path.isfile(write_path), f'File not found, {write_path}'
write.to_csv(write_path, mode='w', header=False, index=False)

# 建立保存测试集的文件夹,并将测试集图片保存到测试集文件夹
col_result = ['img_path', 'LogImages_path', 'score', 'style', 'grade']
result_data = []
test_images_root = os.path.join(save_DistortedImages_dir, "test")
mk_file(test_images_root)
test_Logimages_root = os.path.join(save_LogImages_dir, "test")
mk_file(test_Logimages_root)
for test_image in test_images:
    # 将分配至训练集中的文件复制到相应目录
    image_path = os.path.join(dataset_DistortedImages_dir, test_image)
    test_images_path.append(image_path)
    Logimage_path = os.path.join(dataset_LogImages_dir, test_image)
    test_LogImages_path.append(Logimage_path)
    test_images_label.append(image_score_dict[test_image])

    Images_new_path = os.path.join(test_images_root, test_image)
    copy(image_path, Images_new_path)
    LogImages_new_path = os.path.join(test_Logimages_root, test_image)
    copy(Logimage_path, LogImages_new_path)
    result_data.append(
        [Images_new_path, LogImages_new_path, image_score_dict[test_image], image_style_dict[test_image],
         image_grade_dict[test_image]])

write = pd.DataFrame(columns=col_result, data=result_data)
write_path = os.path.join(os.path.dirname(__file__), 'test.csv')
assert os.path.isfile(write_path), f'File not found, {write_path}'
write.to_csv(write_path, mode='w', header=False, index=False)

# train+val
write = pd.DataFrame(columns=col_result, data=total_data)
write_path = os.path.join(os.path.dirname(__file__), 'train_val.csv')
assert os.path.isfile(write_path), f'File not found, {write_path}'
write.to_csv(write_path, mode='w', header=False, index=False)
