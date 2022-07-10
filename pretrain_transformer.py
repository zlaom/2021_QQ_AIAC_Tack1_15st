import pandas as pd
import yaml
import os
import torch
import logging
import scipy
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from sklearn.model_selection import train_test_split
from dataset.my_dataset import IdDataset, IndexDataset
from model.models import MyBertTrsSeModel, MyAttentionModel, MySingleBertModel, MySingleNextVladModel, MySingleTrsModel
from model.bmt.my_bmt import MyBmtDemo1
from utils.utils import PRScore
from tqdm import tqdm
from model.loss import MutilLabelSmoothing
from model.finetune_model import MyRetrainMode2
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

# %%
logging.info('读取配置文件')
yamlPath = "config/trs_pretrain_config.yaml"
f = open(yamlPath, 'r', encoding='utf-8')
config = f.read()
config = yaml.load(config)

dataset_cfg = config['DATASET']
model_cfg = config['MODEL']
# model_cfg["NAME"] = "CONCAT_W_DROP2_10000"

# 必要文件路径
pointwise_info_path = dataset_cfg['POINTWISE']['INFO_PATH']
pairwise_info_path = dataset_cfg['PAIRWISE']['INFO_PATH']
pairwise_label_path = dataset_cfg['PAIRWISE']['LABEL_FILE']

logging.info('构建dataloader')
# train eval dataloader 使用所有数据的
pointwise_info = pd.read_csv(pointwise_info_path)
pairwise_info = pd.read_csv(pairwise_info_path)

train_id_list = pointwise_info.index.tolist()
val_id_list = pairwise_info.index.tolist()

logging.info('train_set: {} val_set:{}'.format(len(train_id_list), len(val_id_list)))

train_id_set = IndexDataset(dataset_cfg, train_id_list, pointwise_info)
val_id_set = IndexDataset(dataset_cfg, val_id_list, pairwise_info)

train_id_loader = DataLoader(dataset=train_id_set,
                             batch_size=config['TRAINING']['BATCH_SIZE'],
                             shuffle=True)
val_id_loader = DataLoader(dataset=val_id_set,
                           batch_size=config['TRAINING']['EAVL_BATCH_SIZE'],
                           shuffle=False)

# spearman eval dataloader
pairwise_info = pd.read_csv(pairwise_info_path)
id_list = pairwise_info['id'].tolist()
logging.info('pointwise_set: {}'.format(len(id_list)))

spearman_id_set = IdDataset(dataset_cfg, id_list, pairwise_info, test=True)

spearman_id_loader = DataLoader(dataset=spearman_id_set,
                                batch_size=config['TRAINING']['EAVL_BATCH_SIZE'],
                                shuffle=False)


@torch.no_grad()
def evaluate_pr(epoch, model, dataloader):
    logging.info('begin evaluate precision recall')
    model.eval()
    metric = PRScore()
    for batch in tqdm(dataloader):
        labels = batch[label].cuda().float()
        input_feature = get_feature(batch, need)
        pred, embedding = model(*input_feature)
        metric.collect(labels, pred)
        # break
    info = metric.calc()
    metric.reset()
    logging.info("epoch:{} evaluate precision:{:.4f} recall:{:.4f}".format(
        epoch, info['precision'], info['recall']))
    model.train()
    return info


@torch.no_grad()
def evaluate_spearman(epoch, model, dataloader, label_file):
    logging.info('begin evaluate spearman')
    model.eval()
    id_list = []
    embedding_list = []
    for batch in tqdm(dataloader):
        ids = batch['id']
        input_feature = get_feature(batch, need)
        pred, embedding = model(*input_feature)
        embedding = embedding.detach().cpu().numpy()
        embedding_list.append(embedding)
        id_list += ids
        # break
    embeddings = np.concatenate(embedding_list)
    embedding_map = dict(zip(id_list, embeddings))
    annotate = {}
    with open(label_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            rk1, rk2, score = line.split('\t')
            annotate[(rk1, rk2)] = float(score)
    sim_res = []
    logging.info('num embedding: {}, num annotates: {}'.format(len(embedding_map), len(annotate)))
    for (k1, k2), v in annotate.items():
        if k1 not in embedding_map or k2 not in embedding_map:
            continue
        sim_res.append((v, (embedding_map[k1] * embedding_map[k2]).sum()))
    spearman = scipy.stats.spearmanr([x[0] for x in sim_res], [x[1] for x in sim_res]).correlation
    logging.info('spearman score: {}'.format(spearman))
    model.train()
    return spearman


def get_feature(batch, need):
    result = []
    for key in need:
        f = batch[key]
        if key == 'title_asr':
            f = list(zip(f[0], f[1]))
        result.append(f)

    return result


logging.info('构建模型')
model = MySingleTrsModel(model_cfg)

frozen_layers = []

for layer in frozen_layers:
    for name, value in layer.named_parameters():
        value.requires_grad = False
params = filter(lambda p: p.requires_grad, model.parameters())

for k, v in model.named_parameters():
    print('{}: {}'.format(k, v.requires_grad))

# need = ["title"]
# need = ["title_asr"]
# need = ["frame_feature"]
need = ["frame_feature", "frame_mask"]
# need = ["frame_feature", "frame_mask", "title_asr"]
# need = ["title", "frame_feature"]

label = "tag_id"
# label = "category_id"

logging.info('构建优化器 损失函数')
torch.cuda.set_device(0)
model.cuda()
parameters = model.get_independent_lr_parameters()
optimizer = optim.Adam(parameters)
scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.9**epoch)

loss_fuc = torch.nn.BCELoss()

logging.info('开始训练')

for epoch in range(config['TRAINING']['EPOCHS']):
    model.train()
    metric = PRScore()
    loss_list = []
    for i, batch in enumerate(train_id_loader):
        model.zero_grad()
        optimizer.zero_grad()
        labels = batch[label].cuda().float()
        input_feature = get_feature(batch, need)
        pred, normed_embedding = model(*input_feature)
        metric.collect(labels, pred)
        loss = loss_fuc(pred, labels)
        loss_list.append(loss.cpu().item())
        loss.backward()
        optimizer.step()
        # scheduler.step()
        if (i + 1) % config['TRAINING']['REPORT_STEPS'] == 0:
            info = metric.calc()
            metric.reset()
            logging.info("step:{} precision:{:.4f} recall:{:.4f} loss:{:.4f}".format(
                i, info['precision'], info['recall'],
                loss.cpu().item()))
            # break
        if (i + 1) % config['TRAINING']['SAVE_STEPS'] == 0:
            spearman = evaluate_spearman(epoch, model, spearman_id_loader, pairwise_label_path)
            torch.save(
                model, 'checkpoints/pretrain/{}_epoch{}_step{}_{:.4f}.bin'.format(
                    model_cfg['NAME'], epoch, i, spearman))
        # break
    scheduler.step()
    metric.reset()
    spearman = evaluate_spearman(epoch, model, spearman_id_loader, pairwise_label_path)
    eval_info = evaluate_pr(epoch, model, val_id_loader)
    torch.save(
        model, 'checkpoints/pretrain/{}_epoch{}_{:.4f}_{:.4f}_{:.4f}.bin'.format(
            model_cfg['NAME'], epoch, eval_info['precision'], eval_info['recall'], spearman))
