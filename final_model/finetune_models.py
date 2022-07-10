import torch
import torch.nn as nn
import numpy as np
import transformers as tfs
import torch.nn.functional as F
from .model_base import TaskBase, ModelBase


class MyConcatSeFustionModel(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.video_linear = nn.Linear(cfg['TRS_INPUT'], cfg['OUPUT_SIZE'])
        self.text_linear = nn.Linear(cfg['BERT_INPUT'], cfg['OUPUT_SIZE'])

        self.fusion_se_1 = nn.Sequential(
            nn.Linear((cfg['TRS_INPUT'] + cfg['BERT_INPUT']), cfg['OUPUT_SIZE']),
            nn.BatchNorm1d(cfg['OUPUT_SIZE']), nn.Sigmoid())
        self.fusion_se_2 = nn.Sequential(
            nn.Linear((cfg['TRS_INPUT'] + cfg['BERT_INPUT']), cfg['OUPUT_SIZE']),
            nn.BatchNorm1d(cfg['OUPUT_SIZE']), nn.Sigmoid())

    def forward(self, trs_embedding, bert_embedding):
        fusion_embedding = torch.cat((trs_embedding, bert_embedding), dim=1)
        trs_embedding = self.video_linear(trs_embedding)
        bert_embedding = self.text_linear(bert_embedding)

        trs_embedding_w = self.fusion_se_1(fusion_embedding)
        bert_embedding_w = self.fusion_se_2(fusion_embedding)

        trs_embedding = torch.nn.functional.normalize(trs_embedding, p=2, dim=1)
        bert_embedding = torch.nn.functional.normalize(bert_embedding, p=2, dim=1)

        embedding = trs_embedding * trs_embedding_w + bert_embedding * bert_embedding_w

        return embedding


class HeroConcatSeFustionModel(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.video_linear = nn.Linear(cfg['TRS_INPUT'], cfg['OUPUT_SIZE'])
        self.text_linear = nn.Linear(cfg['BERT_INPUT'], cfg['OUPUT_SIZE'])
        self.hero_linear = nn.Linear(cfg['BERT_INPUT'], cfg['OUPUT_SIZE'])

        self.fusion_se_1 = nn.Sequential(
            nn.Linear((cfg['TRS_INPUT'] + cfg['BERT_INPUT'] * 2), cfg['OUPUT_SIZE']),
            nn.BatchNorm1d(cfg['OUPUT_SIZE']), nn.Sigmoid())
        self.fusion_se_2 = nn.Sequential(
            nn.Linear((cfg['TRS_INPUT'] + cfg['BERT_INPUT'] * 2), cfg['OUPUT_SIZE']),
            nn.BatchNorm1d(cfg['OUPUT_SIZE']), nn.Sigmoid())
        self.fusion_se_3 = nn.Sequential(
            nn.Linear((cfg['TRS_INPUT'] + cfg['BERT_INPUT'] * 2), cfg['OUPUT_SIZE']),
            nn.BatchNorm1d(cfg['OUPUT_SIZE']), nn.Sigmoid())

    def forward(self, trs_embedding, bert_embedding, hero_embedding):
        fusion_embedding = torch.cat((trs_embedding, bert_embedding, hero_embedding), dim=1)
        trs_embedding = self.video_linear(trs_embedding)
        bert_embedding = self.text_linear(bert_embedding)
        hero_embedding = self.hero_linear(hero_embedding)

        trs_embedding_w = self.fusion_se_1(fusion_embedding)
        bert_embedding_w = self.fusion_se_2(fusion_embedding)
        hero_embedding_w = self.fusion_se_3(fusion_embedding)

        trs_embedding = torch.nn.functional.normalize(trs_embedding, p=2, dim=1)
        bert_embedding = torch.nn.functional.normalize(bert_embedding, p=2, dim=1)
        hero_embedding = torch.nn.functional.normalize(hero_embedding, p=2, dim=1)

        embedding = trs_embedding * trs_embedding_w + bert_embedding * bert_embedding_w + hero_embedding * hero_embedding_w

        return embedding


class MyFinetuneModel2(TaskBase):
    def __init__(self, cfg, tag_trs, tag_bert):
        super().__init__(cfg)
        self.tag_trs = tag_trs
        self.tag_bert = tag_bert
        self.fustion = MyConcatSeFustionModel(cfg['MySeFustionModel'])
        self.model_list = [self.fustion]

    def forward(self, frame_feature, mask, asr_title):
        frame_feature = frame_feature.cuda()
        mask = mask.cuda()

        _, tag_trs_embedding = self.tag_trs(frame_feature, mask)
        _, tag_bert_embedding = self.tag_bert(asr_title)

        embedding = self.fustion(tag_trs_embedding, tag_bert_embedding)
        normed_embedding = nn.functional.normalize(embedding, p=2, dim=1)

        b, n = normed_embedding.shape
        embedding_1, embedding_2 = normed_embedding[:b // 2], normed_embedding[b // 2:]

        cos = torch.mul(embedding_1, embedding_2)
        cos = torch.sum(cos, dim=1)

        return cos, normed_embedding


class MyFinetuneModel3(TaskBase):
    def __init__(self, cfg, tag_trs, tag_bert, hero):
        super().__init__(cfg)
        self.tag_trs = tag_trs
        self.tag_bert = tag_bert
        self.hero = hero
        self.fustion = HeroConcatSeFustionModel(cfg['HeroConcatSeFustionModel'])
        self.model_list = [self.fustion]
        self.tokenizer = tfs.BertTokenizer.from_pretrained(cfg['TOKENIZER_PATH'])
        self.CLS = 102

    def forward(self, frame_feature, mask, asr_title, title):
        frame_feature = frame_feature.cuda()
        mask = mask.cuda()

        _, tag_trs_embedding = self.tag_trs(frame_feature, mask)
        _, tag_bert_embedding = self.tag_bert(asr_title)

        batch = {}
        tokenized = self.tokenizer(title,
                                   add_special_tokens=False,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=32,
                                   return_tensors="pt")
        input_ids = tokenized['input_ids']
        title_mask = tokenized['attention_mask']
        b = input_ids.shape[0]
        input_ids = torch.cat((torch.tensor([self.CLS] * b).reshape(-1, 1), input_ids), dim=-1)
        title_mask = F.pad(title_mask.flatten(1), (1, 0), value=1)
        batch['only_input_ids'] = input_ids
        batch['only_title_mask'] = title_mask
        batch['frame_feature'] = frame_feature
        batch['frame_mask'] = mask
        hero_embedding = self.hero(batch, 'repr')[:, 0]

        embedding = self.fustion(tag_trs_embedding, tag_bert_embedding, hero_embedding)
        normed_embedding = nn.functional.normalize(embedding, p=2, dim=1)

        b, n = normed_embedding.shape
        cos = 0
        if self.training:
            embedding_1, embedding_2 = normed_embedding[:b // 2], normed_embedding[b // 2:]

            cos = torch.mul(embedding_1, embedding_2)
            cos = torch.sum(cos, dim=1)

        return cos, normed_embedding


class MyFinetuneModel1(TaskBase):
    def __init__(self, cfg, tag_trs, tag_bert, cat_trs, cat_bert):
        super().__init__(cfg)
        self.tag_trs = tag_trs
        self.tag_bert = tag_bert
        self.cat_trs = cat_trs
        self.cat_bert = cat_bert

        self.fustion = MyConcatSeFustionModel(cfg['MySeFustionModel'])
        self.model_list = [self.fustion]

    def forward(self, frame_feature, mask, asr_title):
        frame_feature = frame_feature.cuda()
        mask = mask.cuda()

        _, tag_trs_embedding = self.tag_trs(frame_feature, mask)
        _, tag_bert_embedding = self.tag_bert(asr_title)
        # _, cat_trs_embedding = self.cat_trs(frame_feature, mask)
        # _, cat_bert_embedding = self.cat_bert(asr_title)

        # tag_trs_hidden = self.tag_trs(frame_feature, mask)
        # tag_trs_embedding = tag_trs_hidden[:, 0]

        # tag_bert_hidden = self.tag_bert(asr_title)
        # tag_bert_embedding = tag_bert_hidden[:, 0]

        cat_trs_hidden = self.cat_trs(frame_feature, mask)
        cat_trs_embedding = cat_trs_hidden[:, 0]

        cat_bert_hidden = self.cat_bert(asr_title)
        cat_bert_embedding = cat_bert_hidden[:, 0]

        video_feature = torch.cat((tag_trs_embedding, cat_trs_embedding), dim=1)
        text_feature = torch.cat((tag_bert_embedding, cat_bert_embedding), dim=1)
        features = torch.cat(
            (tag_trs_embedding, tag_bert_embedding, cat_trs_embedding, cat_bert_embedding), dim=1)
        embedding = self.fustion(tag_trs_embedding, tag_bert_embedding)
        normed_embedding = nn.functional.normalize(embedding, p=2, dim=1)

        b, n = normed_embedding.shape
        embedding_1, embedding_2 = normed_embedding[:b // 2], normed_embedding[b // 2:]

        cos = torch.mul(embedding_1, embedding_2)
        cos = torch.sum(cos, dim=1)

        return cos, normed_embedding