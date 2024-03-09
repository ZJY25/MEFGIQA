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


dataset_root = '/data2/cyk/dataset/SCID/'
dataset_reference_dir = os.path.join(dataset_root, 'ReferenceSCIs')
scores_csv = os.path.join(dataset_root, 'image_labeled_by_score_SCID.csv')
csv_data = pd.read_csv(scores_csv, sep=',', usecols=['image', 'mos', 'style'])
image_score_dict = {}
image_style_dict = {}
for i in range(len(csv_data)):
    image_score_dict[csv_data.iloc[i].image] = csv_data.iloc[i].mos  # images->dmos
    image_style_dict[csv_data.iloc[i].image] = csv_data.iloc[i].style  # images->style
images = list(image_score_dict)

# train_tag, val_tag, test_tag = get_images('images4.csv')
train_tag = ['SCI01', 'SCI02', 'SCI03', 'SCI06', 'SCI07', 'SCI12', 'SCI13', 'SCI14', 'SCI18', 'SCI19', 'SCI20', 'SCI21',
             'SCI22', 'SCI23', 'SCI25', 'SCI26', 'SCI28', 'SCI29', 'SCI32', 'SCI33', 'SCI34', 'SCI37', 'SCI38', 'SCI39']
val_tag = ['SCI04', 'SCI05', 'SCI10', 'SCI17', 'SCI24', 'SCI35', 'SCI31', 'SCI40']
test_tag = ['SCI08', 'SCI09', 'SCI11', 'SCI15', 'SCI16', 'SCI27', 'SCI30', 'SCI36']
# 0:['SCI08', 'SCI09', 'SCI11', 'SCI15', 'SCI16', 'SCI24', 'SCI27', 'SCI36']
# 20:['SCI08', 'SCI09', 'SCI11', 'SCI15', 'SCI16', 'SCI24', 'SCI27', 'SCI35']
# 18:['SCI08', 'SCI09', 'SCI11', 'SCI15', 'SCI16', 'SCI24', 'SCI30', 'SCI31']

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
write_path = os.path.join(os.path.dirname(__file__), 'image24.csv')
print(write_path)
assert os.path.isfile(write_path), f'File not found, {dataset_root}'
write.to_csv(write_path, index=False, header=False)
