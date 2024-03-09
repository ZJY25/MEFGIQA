import csv
import os
import pandas as pd
from itertools import islice


def Data_Preprocessing(epoch):
    select_epoch = epoch  # 选择指定的epoch轮数
    file = open('SIQAD_TestResult.csv', 'r')
    lines = file.readlines()
    file.close()
    row = []

    for line in lines:
        line.strip()
        row.append(line.split(','))

    result_data = []
    col_result = ['epoch', 'test_images', 'pred_score', 'gt_score']

    for k in range(1, 8):
        for c in row:
            if c[0] == select_epoch and c[1].split('_')[1] == str(k):
                c[3] = c[3].replace('\n', '')
                result_data.append(c)

    write = pd.DataFrame(columns=col_result, data=result_data)
    write_path = os.path.join(os.path.dirname(__file__), '{}_TestResult.csv'.format(select_epoch))
    if os.path.exists(write_path):
        os.remove(write_path)
    else:
        file = open(write_path, 'w')
        file.close()
    write.to_csv(write_path, mode='a', header=True, index=False)
    return write_path


def Data_Caculate(filePath):

    for k in range(1, 8):
        with open('./test_result/result_{}.txt'.format(k), 'w') as file_handle:  # .txt可以不自己新建,代码会自动新建
            file_handle.write("{}\n".format("predscore" + " " + "GT"))
        with open(filePath, 'r', encoding="utf-8") as csvfile:
            reader = csv.reader(csvfile)
            for row in islice(reader, 1, None):
                if row[1].split('_')[1] == str(k):
                    with open('./test_result/result_{}.txt'.format(k), 'a') as file_handle:
                        file_handle.write("{}\n".format(str(row[2]) + " " + str(row[3])))
    csvfile.close()
    file_handle.close()



if __name__ == "__main__":
    epoch = "43"
    file_path = Data_Preprocessing(epoch)
    Data_Caculate(file_path)