import torch
import json
import numpy as np
import pandas as pd
import yaml
import os
import torch
import logging
import numpy as np
from torch.utils.data import DataLoader
from dataset.my_dataset import TestIndexDataset
import model.models
from tqdm import tqdm
import zipfile
import re

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

# %%
logging.info('读取配置文件')
yamlPath = "config/test.yaml"
f = open(yamlPath, 'r', encoding='utf-8')
config = f.read()
config = yaml.load(config)

dataset_cfg = config['DATASET']
testing_cfg = config['TESTING']

# 必要文件路径
test_info_path = dataset_cfg['TEST_B']['INFO_PATH']

test_info = pd.read_csv(test_info_path)
index_list = test_info.index.tolist()
logging.info('test_set: {}'.format(len(index_list)))
test_id_set = TestIndexDataset(dataset_cfg, index_list, test_info)
test_id_loader = DataLoader(dataset=test_id_set,
                            batch_size=config['TESTING']['BATCH_SIZE'],
                            shuffle=False)


def get_feature(batch, need):
    result = []
    for key in need:
        f = batch[key]
        if key == 'title_asr':
            f = list(zip(f[0], f[1]))
        result.append(f)

    return result


@torch.no_grad()
def test_checkpoint(model, dataloader, to_save_file):
    """
    :param checkpoint_file: the checkpoint used to evaluate the dataset
    :param dataset_key: dataset indicator key, defined by your_config.yaml DATASET block
    :param to_save_file: the file to save the result
    :return:
    """
    logging.info('开始生成embedding')
    model.eval()
    id_list = []
    embedding_list = []
    for batch in tqdm(dataloader):
        input_feature = get_feature(batch, need)
        ids = batch['id']
        pred, embedding = model(*input_feature)
        embedding = embedding.detach().cpu().numpy()
        id_list.extend(list(ids))
        embedding_list.extend(embedding.astype(np.float16).tolist())
    output_res = dict(zip(id_list, embedding_list))
    with open(to_save_file, 'w') as f:
        json.dump(output_res, f)
    logging.info('结果文件保存至:{}'.format(to_save_file))


if os.path.exists(testing_cfg['SAVE_PATH']):
    logging.info('删除上一结果文件:{}'.format(testing_cfg['SAVE_PATH']))
    os.remove(testing_cfg['SAVE_PATH'])

need = ["frame_feature", "frame_mask", "title_asr", "title"]
logging.info(testing_cfg['CHECKPOINT'])
model = torch.load(testing_cfg['CHECKPOINT'])
# model = model.pretrain
model.cuda()

test_checkpoint(model, test_id_loader, testing_cfg['SAVE_PATH'])

# 压缩文件
save_path = testing_cfg['CHECKPOINT']
save_path = re.sub(r".bin$", "", save_path)
save_path = re.sub(r"0\.", "", save_path)
save_path = re.sub(r"^.+/", "result/", save_path)
save_path = '{}.zip'.format(save_path)

with zipfile.ZipFile(save_path, 'w', zipfile.ZIP_DEFLATED) as myzip:
    myzip.write(testing_cfg['SAVE_PATH'], arcname="result.json")
logging.info('压缩文件保持至:{}'.format(save_path))