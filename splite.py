import json
import random
import os 
# 假设你的json文件名为data.json
def split(path):
    with open(os.path.join(path,"val_1_shot_knn.json"), 'r') as file:
        data = json.load(file)
    total_samples = len(data)
    split_10_percent = int(total_samples * 0.1) # 10%
    # 随机打乱数据顺序
    random.shuffle(data)

    # 分割数据
    split_10_data = data[:split_10_percent]
    split_5_data = split_10_data[:int(len(split_10_data)*0.5)]
    split_90_data = data[split_10_percent:]

    save_json(split_10_data, path+'/knn_train10%.json')
    save_json(split_90_data, path+'/knn_test90%.json')
    save_json(split_5_data, path+'/knn_train5%.json')
def save_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
path = "/home/DATA2/cxh/Eval_data"
split(path)
