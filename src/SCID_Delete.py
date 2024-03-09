import shutil
import os


def clear_folder(folder_path):
    # 使用shutil模块的rmtree函数删除文件夹及其内部的所有文件
    shutil.rmtree(folder_path)

    # 使用os模块的makedirs函数重新创建一个同名的空文件夹
    os.makedirs(folder_path)

if __name__=="__main__":
    with open("SCID_ScoreData.csv", "r+") as f:
        f.seek(0)  # 移动文件指针到文件开头
        f.truncate()  # 清空文件内容

    with open("SCID_TestResult.csv", "r+") as f:
        f.seek(0)  # 移动文件指针到文件开头
        f.truncate()  # 清空文件内容

    with open("Case_SCID.log", "r+") as f:
        f.seek(0)  # 移动文件指针到文件开头
        f.truncate()  # 清空文件内容

    folder_path = "/data2/cyk/MyIQA2New/src/model_result/SCID/test"  # 指定文件夹路径
    clear_folder(folder_path)  # 清空文件夹内的所有文件

    folder_path = "/data2/cyk/MyIQA2New/src/model_result/SCID/val"  # 指定文件夹路径
    clear_folder(folder_path)  # 清空文件夹内的所有文件