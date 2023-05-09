# 作者：钱隆
# 时间：2023/5/10 1:00


import csv

# 读取list_bbox_celeba.txt文件，获取坐标信息
with open('list_bbox_celeba.txt', 'r') as f:
    reader = csv.reader(f, delimiter=' ')
    # 跳过前两行注释和列名
    next(reader)
    next(reader)
    # 依次处理每一行数据，将坐标信息按照对应的id保存到对应文件夹中
    for row in reader:
        # 获取文件id和坐标信息
        file_id = row[0]
        x_1, y_1, width, height = map(int, row[1:])
        x_2 = x_1 + width
        y_2 = y_1 + height
        # 确定文件所属的集合
        if file_id in train_set:
            subset = 'train'
        elif file_id in val_set:
            subset = 'val'
        else:
            subset = 'test'
        # 拼接文件路径
        file_path = f'{subset}/{file_id}/{file_id}.txt'
        # 打开文件，将坐标信息写入
        with open(file_path, 'w') as f:
            f.write(f'{file_id},{x_1},{y_1},{x_2},{y_2}\n')