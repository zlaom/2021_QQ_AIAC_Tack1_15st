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
from dataset.my_dataset import IdDataset, PairwiseDataset
from model.models import MySingleTrsModel
from model.my_mlm import MyBmtMLM
from model.my_cross_trs import MyCrossFinetune2, MyCrossFinetune
from final_model.finetune_models import MyFinetuneModel3
from utils.utils import PRScore
from utils.model_load import model_parameters_load
from model.loss import CrossMSE, FusionCrossMSE, DistMSE
from tqdm import tqdm
from transformers import (AutoConfig, AutoTokenizer, AutoModelForMaskedLM, AdamW,
                          get_linear_schedule_with_warmup)

from model.hero.model import VideoModelConfig
from model.hero.my_model import HierarchicalVlModel

hero_config = VideoModelConfig('config/hero_pretrain.json')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
torch.cuda.set_device(0)


@torch.no_grad()
def evaluate_pr(epoch, model, dataloader):
    logging.info('begin evaluate precision recall')
    model.eval()
    metric = PRScore()
    for batch in tqdm(train_loader, ncols=60):
        frame_feature = batch["frame_feature"].cuda()
        labels = batch["tag_id"].cuda().float()
        pred, embedding = model(frame_feature)
        metric.collect(labels, pred)
        # break
    info = metric.calc()
    metric.reset()
    logging.info("epoch:{} evaluate precision:{:.4f} recall:{:.4f}".format(
        epoch, info['precision'], info['recall']))
    model.train()
    return info


def formate_change(x):
    return list(zip(x[0], x[1]))


@torch.no_grad()
def evaluate_spearman(epoch, model):
    model.eval()
    logging.info('begin evaluate spearman epoch')
    cos_list = []
    score_list = []
    for record in tqdm(val_loader, ncols=40):
        title_1, video_features_1, mask1 = record['title1'], record['video_feature1'], record[
            'mask1']
        title_2, video_features_2, mask2 = record['title2'], record['video_feature2'], record[
            'mask2']
        title_asr_1, title_asr_2 = formate_change(record['title_asr1']), formate_change(
            record['title_asr2'])
        scores = record['score']
        title = np.concatenate((title_1, title_2), axis=0)
        title = title.tolist()

        video_features = torch.cat((video_features_1, video_features_2), axis=0)
        mask = torch.cat((mask1, mask2), axis=0)
        title_asr = []
        title_asr.extend(title_asr_1)
        title_asr.extend(title_asr_2)

        video_features = video_features.cuda()
        mask = mask.cuda()

        cos, embedding = model(video_features, mask, title_asr, title)

        cos = cos.cpu().numpy()
        cos_list.append(cos)
        score_list.append(scores)

    cos_list = np.concatenate(cos_list)
    score_list = np.concatenate(score_list)

    spearman = scipy.stats.spearmanr([x for x in cos_list], [x for x in score_list]).correlation
    logging.info('epoch {} eval spearman: {:.4f}'.format(epoch, spearman))
    return spearman


def batch_spearman(cos, scores):
    sim_res = cos.detach().cpu().numpy()
    scores = scores.detach().cpu().numpy()
    spearman = scipy.stats.spearmanr(sim_res, scores).correlation
    return spearman


def get_feature(batch, need):
    result = []
    for key in need:
        result.append(batch[key])
    return result


def get_all_spearman(cos_list, score_list):
    cos_list = np.concatenate(cos_list)
    score_list = np.concatenate(score_list)
    spearman = scipy.stats.spearmanr([x for x in cos_list], [x for x in score_list]).correlation
    return spearman


def train(epoch):

    loss_list = []
    cos_list = []
    score_list = []
    for i, record in enumerate(train_loader):
        model.train()
        optimizer.zero_grad()

        title_1, video_features_1, mask1 = record['title1'], record['video_feature1'], record[
            'mask1']
        title_2, video_features_2, mask2 = record['title2'], record['video_feature2'], record[
            'mask2']
        title_asr_1, title_asr_2 = formate_change(record['title_asr1']), formate_change(
            record['title_asr2'])
        scores = record['score']
        title = np.concatenate((title_1, title_2), axis=0)
        title = title.tolist()

        video_features = torch.cat((video_features_1, video_features_2), axis=0)
        mask = torch.cat((mask1, mask2), axis=0)
        title_asr = []
        title_asr.extend(title_asr_1)
        title_asr.extend(title_asr_2)

        video_features = video_features.cuda()
        scores = scores.cuda().float()
        mask = mask.cuda()

        cos, embedding = model(video_features, mask, title_asr, title)

        cos_list.append(cos.detach().cpu().numpy())
        score_list.append(scores.detach().cpu().numpy())
        loss = loss_fn(cos, scores)
        # normal_cos = torch.nn.functional.normalize(cos, p=2, dim=0)
        # normal_scores = torch.nn.functional.normalize(scores, p=2, dim=0)
        # loss = loss_fn(normal_cos, normal_scores)

        loss_list.append(loss.cpu().item())

        loss.backward()
        optimizer.step()
        # scheduler.step()
        # break
        if (i + 1) % config['FINETUNE']['REPORT_STEPS'] == 0:
            spearman = get_all_spearman(cos_list, score_list)
            logging.info('The epoch of {} step: {} train spearman: {:.4f} loss: {:.4f}'.format(
                epoch, i, spearman,
                loss.cpu().item()))


if __name__ == "__main__":
    logging.info('读取配置文件')
    yaml_path = "config/hero_finetune_config.yaml"
    f = open(yaml_path, 'r', encoding='utf-8')
    config = f.read()
    config = yaml.load(config)

    dataset_cfg = config['DATASET']
    hero_cfg = config['FINETUNE']

    pairwise_info_path = dataset_cfg['PAIRWISE']['INFO_PATH']
    # pairwise_label_path = dataset_cfg['PAIRWISE']['LABEL_FILE']

    train_tsv_path = dataset_cfg['PAIRWISE']['TRAIN_PATH']
    val_tsv_path = dataset_cfg['PAIRWISE']['VAL_PATH']

    logging.info('构建dataloader')
    # train eval dataloader
    pairwise_info = pd.read_csv(pairwise_info_path)

    train_dataset = PairwiseDataset(dataset_cfg, train_tsv_path, pairwise_info)
    val_dataset = PairwiseDataset(dataset_cfg, val_tsv_path, pairwise_info)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=config['FINETUNE']['BATCH_SIZE'],
                              shuffle=True)
    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=config['FINETUNE']['BATCH_SIZE'],
                            shuffle=False)

    logging.info('构建model')

    hero = HierarchicalVlModel(hero_config)
    hero_path = hero_cfg['HERO']
    hero.load_state_dict(torch.load(hero_path))

    tag_trs = torch.load(hero_cfg['TAG_FRAME_TRS_CONV']).trs
    tag_bert = torch.load(hero_cfg['TAG_TITLE_ASR_BERT']).bert
    model = MyFinetuneModel3(hero_cfg, tag_trs, tag_bert, hero)

    model.cuda()
    frozen_layers = [model.tag_trs, model.tag_bert, model.hero]

    for layer in frozen_layers:
        for name, value in layer.named_parameters():
            value.requires_grad = False
    params = filter(lambda p: p.requires_grad, model.parameters())

    # parameters = model.get_independent_lr_parameters()
    # for item in parameters:
    #     if item['lr'] > config['FINETUNE']['LEARNING_RATE']:
    #         item['lr'] = config['FINETUNE']['LEARNING_RATE']
    # parameters[-1]['lr'] = 1e-4
    for k, v in model.named_parameters():
        print('{}: {}'.format(k, v.requires_grad))

    optimizer = optim.Adam(model.parameters(),
                           lr=1e-4,
                           weight_decay=config['FINETUNE']['WEIGHT_DECAY'])
    # optimizer = AdamW(model.parameters(), lr=config['FINETUNE']['LEARNING_RATE'], eps=1e-8)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # scheduler = LambdaLR(optimizer=optimizer, lr_lambda=lambda epoch: 0.9**epoch)
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=0,
    #     num_training_steps=int(len(train_loader) * config['TRAINING']['EPOCHS']))
    loss_fn = torch.nn.MSELoss(reduction='mean')

    logging.info('开始训练')
    for epoch in range(config['FINETUNE']['EPOCHS']):
        train(epoch)
        spearman = evaluate_spearman(epoch, model)
        torch.save(
            model,
            'checkpoints/finetune/{}_epoch{}_{:.4f}.bin'.format(hero_cfg['NAME'], epoch, spearman))
        # scheduler.step()
