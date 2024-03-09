import os
import random
import csv
import pandas as pd


def function(data):
    return int(data.split('cim')[-1])


# 读取images.csv文件中的每一行数据,并存入相应的训练集、验证集和测试集
total_images=[]
read_path=os.path.join(os.path.dirname(__file__), 'image31.csv')
with open(read_path,'r',encoding="utf-8") as csvfile:
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
train_images.sort(key=function)
val_images = list(set(val_images))
val_images.sort(key=function)
test_images = list(set(test_images))
test_images.sort(key=function)

print(train_images)
print(val_images)
print(test_images)