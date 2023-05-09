# 作者：钱隆
# 时间：2023/5/10 0:25

import os
import shutil

# 读取 list_eval_partition.txt 文件，解析出图像文件名和数据集类型的对应关系
partition_file = 'list_eval_partition.txt'
partition_dict = {}
with open(partition_file, 'r') as f:
    for line in f:
        img_name, partition_type = line.strip().split()
        partition_dict[img_name] = partition_type

# 读取 identity_CelebA.txt 文件，解析出每个人物 ID 对应的图像文件名
identity_file = 'identity_CelebA.txt'
identity_dict = {}
with open(identity_file, 'r') as f:
    for line in f:
        img_name, identity_id = line.strip().split()
        identity_dict[img_name] = identity_id

# 划分数据集
data_folder = 'img_celeba'
train_folder = 'train'
val_folder = 'val'
test_folder = 'test'
for img_name in os.listdir(data_folder):
    # 确定数据集类型
    partition_type = partition_dict[img_name]
    if partition_type == '0':
        target_folder = train_folder
    elif partition_type == '1':
        target_folder = val_folder
    else:
        target_folder = test_folder

    # 确定人物 ID
    identity_id = identity_dict[img_name]

    # 创建目标文件夹（如果不存在）
    target_path = os.path.join(target_folder, identity_id)
    os.makedirs(target_path, exist_ok=True)

    # 复制图像文件到目标文件夹
    img_path = os.path.join(data_folder, img_name)
    shutil.copy(img_path, target_path)
