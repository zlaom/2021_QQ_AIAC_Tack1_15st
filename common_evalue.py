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
from model.finetune_model import MyFinetuneModel1, MyFinetuneModel4, MyFinetuneModel2
from utils.utils import PRScore
from utils.model_load import model_parameters_load
from model.loss import CrossMSE, FusionCrossMSE
from tqdm import tqdm

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s %(message)s',
                    datefmt='%a, %d %b %Y %H:%M:%S')

os.environ["CUDA_VISIBLE_DEVICES"] = "5"
torch.cuda.set_device(0)


def formate_change(x):
    return list(zip(x[0], x[1]))


@torch.no_grad()
def finetune__evaluate_spearman(epoch, model):
    model.eval()
    logging.info('begin evaluate spearman epoch')
    cos_list = []
    score_list = []
    for record in tqdm(finetune_val_loader, ncols=40):
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
        cos, embedding = model(video_features, mask, title_asr, title)
        cos = cos.cpu().numpy()
        cos_list.append(cos)
        score_list.append(scores)

    cos_list = np.concatenate(cos_list)
    score_list = np.concatenate(score_list)

    spearman = scipy.stats.spearmanr([x for x in cos_list], [x for x in score_list]).correlation
    logging.info('epoch {} eval spearman: {:.4f}'.format(epoch, spearman))
    return spearman


@torch.no_grad()
def pretrain_evaluate_spearman(model, dataloader, label_file):
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


if __name__ == "__main__":
    logging.info('读取配置文件')
    yaml_path = "config.yaml"
    f = open(yaml_path, 'r', encoding='utf-8')
    config = f.read()
    config = yaml.load(config)

    dataset_cfg = config['DATASET']
    model_cfg = config['MODEL']
    pairwise_label_path = dataset_cfg['PAIRWISE']['LABEL_FILE']

    logging.info('构建dataloader')
    pairwise_info_path = dataset_cfg['PAIRWISE']['INFO_PATH']
    pairwise_info = pd.read_csv(pairwise_info_path)
    id_list = pairwise_info['id'].tolist()
    logging.info('pointwise_set: {}'.format(len(id_list)))

    val_tsv_path = dataset_cfg['PAIRWISE']['VAL_PATH']

    finetune_val_dataset = PairwiseDataset(dataset_cfg, val_tsv_path, pairwise_info)
    finetune_val_loader = DataLoader(dataset=finetune_val_dataset,
                                     batch_size=config['FINETUNE']['BATCH_SIZE'],
                                     shuffle=False)

    spearman_id_set = IdDataset(dataset_cfg, id_list, pairwise_info, test=True)
    spearman_id_loader = DataLoader(dataset=spearman_id_set,
                                    batch_size=config['TESTING']['BATCH_SIZE'],
                                    shuffle=False)

    logging.info('构建model')
    # tag trs + bert
    # model = torch.load(config['FINETUNE']['TAG_FRAME_TRS'])
    # model = torch.load(config['FINETUNE']['CAT_FRAME_TRS'])
    # model = torch.load(config['FINETUNE']['TAG_FRAME_TRS_CONV'])

    # model = torch.load(config['FINETUNE']['TAG_NEXTVLAD'])
    # model = torch.load(config['FINETUNE']['TAG_TITLE_ASR_BERT'])
    # model = torch.load(config['FINETUNE']['CAT_TITLE_ASR_BERT'])
    # model = MyFinetuneModel1(model_cfg, tag_trs, tag_bert)

    model = torch.load('checkpoints/finetune/HERO12_FINTUNE_SCORE_CLIP_epoch18_0.7958.bin')
    model.cuda()

    label = "tag_id"
    # label = "category_id"

    # need = ["title"]
    # need = ["title_asr"]
    # need = ["frame_feature"]
    # need = ["frame_feature", "frame_mask"]
    need = ["frame_feature", "frame_mask", "title_asr", "title"]
    # need = ["title", "frame_feature"]

    logging.info('开始评估')
    # spearman = pretrain_evaluate_spearman(model, spearman_id_loader, pairwise_label_path)
    spearman = finetune__evaluate_spearman(0, model)
    logging.info('spearman: {}'.format(spearman))
