import os
import random
import csv
import pandas as pd


# 读取images.csv文件中的每一行数据,并存入相应的训练集、验证集和测试集
total_images = []
read_path = os.path.join(os.path.dirname(__file__), 'image22.csv')
with open(read_path, 'r', encoding="utf-8") as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]
for i in range(len(rows)):
    total_images.append([x for x in rows[i] if x != ''])

train_images = total_images[0]
val_images = total_images[2]
test_images = total_images[1]

# 将训练集、验证集和测试集中的图片名按行存入images.csv文件
images_list=[]
images_list.append(train_images)
images_list.append(val_images)
images_list.append(test_images)
write=pd.DataFrame(data=images_list)
write_path=os.path.join(os.path.dirname(__file__), 'image22.csv')
print(write_path)
assert os.path.isfile(write_path), f'File not found, {write_path}'
write.to_csv(write_path,index=False,header=False)
