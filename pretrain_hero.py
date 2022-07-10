import pandas as pd
import yaml
import os
import torch
import torch.nn.functional as F
import logging
import scipy
import numpy as np
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from sklearn.model_selection import train_test_split
from dataset.hero_dataset import TagValDataset, VsmDataset

from tqdm import tqdm
from utils.model_saver import ModelSaver
# bert encoder
from model.hero.model import VideoModelConfig
from model.hero.my_model import HierarchicalVlModel
from dataset.metaloader import MetaLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
torch.cuda.set_device(0)
saving_dir = 'checkpoints/pretrain'
saving_name = 'hero'
spearman_model_saver = ModelSaver(saving_dir, saving_name + '_spearm')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

# config
logging.info('读取配置文件')
yamlPath = "config/hero_pretrain_config.yaml"
f = open(yamlPath, 'r', encoding='utf-8')
config = f.read()
config = yaml.load(config)
dataset_cfg = config['DATASET']
model_cfg = config['MODEL']

hero_config = VideoModelConfig('config/hero_pretrain.json')

# model construction
logging.info('构建模型')
model = HierarchicalVlModel(hero_config)
# dict_path = 'checkpoints/state_dict/pretrain/mlm_mfm_vsm/6l_trs_heroptim_mfm_epoch30_0.7168'
# model.load_state_dict(torch.load(dict_path), strict=False)
model.cuda()

# tools for training model
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {
        'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        'lr': 3e-5,
        'weight_decay': 0.01
    },
    {
        'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        'lr': 3e-5,
        'weight_decay': 0.0
    },
]
optimizer = optim.AdamW(optimizer_grouped_parameters, lr=3e-5, betas=[0.9, 0.98])

# optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=[0.9, 0.98])

# dataset and dataloader
logging.info('构建dataloader')
pairwise_info_path = dataset_cfg['PAIRWISE']['INFO_PATH']
pairwise_label_path = dataset_cfg['PAIRWISE']['LABEL_FILE']
pointwise_info_path = dataset_cfg['POINTWISE']['INFO_PATH']
pointwise_info = pd.read_csv(pointwise_info_path)

# split for pointwise data
id_list = pointwise_info.index.tolist()
train_id_list, val_id_list = train_test_split(id_list, test_size=0.05, random_state=42)
logging.info('train_set: {} val_set:{}'.format(len(train_id_list), len(val_id_list)))
# spearman
pairwise_info = pd.read_csv(pairwise_info_path)
id_list = pairwise_info['id'].tolist()

tag_train_id_set = VsmDataset(dataset_cfg, train_id_list, pointwise_info)
spearman_id_set = TagValDataset(dataset_cfg, id_list, pairwise_info, test=True)

tag_train_id_loader = DataLoader(dataset=tag_train_id_set,
                                 batch_size=config['TRAINING']['BATCH_SIZE'],
                                 shuffle=True,
                                 drop_last=True)
spearman_id_loader = DataLoader(dataset=spearman_id_set,
                                batch_size=config['TESTING']['BATCH_SIZE'],
                                shuffle=False)

loaders = {'tag': tag_train_id_loader}
meta_loader = MetaLoader(loaders)


# evaluate mlm
@torch.no_grad()
def validate_mlm(model, val_loader):
    val_loss = 0
    n_correct = 0
    n_word = 0

    for i, batch in tqdm(enumerate(val_loader)):
        scores, labels = model(batch, task='mlm', compute_loss=False)
        loss = F.cross_entropy(scores, labels, reduction='mean')
        val_loss += loss.item()
        n_correct += (scores.max(dim=-1)[1] == labels).sum().item()
        n_word += labels.numel()

    val_loss /= n_word
    acc = n_correct / n_word
    print(f"val loss: {val_loss:.4f}", f"acc: {acc*100:.2f}%")
    return acc


# evaluate mfm
@torch.no_grad()
def validate_mfm(model, val_loader):
    val_loss = 0
    n_correct = 0
    cosine = 0
    n_feat = 0
    n_neg = 0
    for i, batch in tqdm(enumerate(val_loader)):
        feats, pos_feats, neg_feats = model(batch, task='mfm', compute_loss=False)
        logits = model.mfm_nce(feats, pos_feats, neg_feats, compute_loss=False)
        targets = torch.arange(0, logits.size(0), dtype=torch.long, device=logits.device)
        val_loss += F.cross_entropy(logits, targets, reduction='sum').item()
        n_correct += (logits.max(dim=-1)[1] == targets).sum().item()
        cosine += F.cosine_similarity(feats, pos_feats, dim=-1).sum().item()
        nf = pos_feats.size(0)
        n_feat += nf
        n_neg += neg_feats.size(0) * nf

    val_loss /= n_feat
    val_acc = n_correct / n_feat
    print(f"val loss: {val_loss:.4f}", f"acc: {val_acc*100:.2f}%",
          f"(average {n_neg/n_feat:.0f} negatives)")
    return val_acc


# evaluate spearman
@torch.no_grad()
def evaluate_spearman(model, val_loader, label_file):
    logging.info('begin evaluate spearman')
    model.eval()
    id_list = []
    embedding_list = []
    for i, batch in tqdm(enumerate(val_loader)):
        ids = batch['id']
        embedding = model(batch, 'tag', compute_loss=False)
        embedding = embedding.detach().cpu().numpy()
        embedding_list.append(embedding)
        id_list += ids
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


# training
logging.info('开始训练')
model.train()

epoch = 0
tasks = ['tag']
loss_list = {}
for task in tasks:
    loss_list[task] = []

report_step = len(tasks) * config['TRAINING']['REPORT_STEPS']
save_step = 2 * len(tasks) * config['TRAINING']['SAVE_STEPS']
for i, (task, batch) in enumerate(meta_loader):
    optimizer.zero_grad()

    loss = model(batch, task)
    loss_list[task].append(loss.cpu().item())
    loss.backward()
    optimizer.step()

    # evaluating PRscore
    if (i + 1) % report_step == 0:
        for task in tasks:
            l = loss_list[task]
            if l:  # 如果不为空
                avg_loss = sum(l) / len(l)
            else:
                avg_loss = -1
            loss_list[task] = []
            print(f'{task}:{avg_loss:.4f};', end='    ')
        print()

    # evaluate spearman
    if (i + 1) % save_step == 0:
        spearman = evaluate_spearman(model, spearman_id_loader, pairwise_label_path)
        spearman_model_saver.save_model(model, spearman, epoch)
        epoch += 1