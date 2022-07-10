import torch
import torch.nn as nn
from model.model_base import TaskBase, ModelBase
from model.nextvlad import NeXtVLAD
from model.bmt.encoder import BiModalEncoder
from model.bmt.blocks import BridgeConnection


class MyBimdalFustionModel(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.bmt = BiModalEncoder(d_model_A=1536,
                                  d_model_V=768,
                                  d_model=None,
                                  dout_p=0.1,
                                  H=8,
                                  d_ff_A=3072,
                                  d_ff_V=1536,
                                  N=2)
        self.bridge = BridgeConnection(2304, 256, 0.1)

    def forward(self, trs_embedding, bert_embedding):
        trs_embedding = torch.unsqueeze(trs_embedding, dim=1)
        bert_embedding = torch.unsqueeze(bert_embedding, dim=1)

        # 重复10次
        trs_embedding = trs_embedding.repeat(1, 10, 1)
        bert_embedding = bert_embedding.repeat(1, 10, 1)

        B, S, D = trs_embedding.shape
        mask = {'A_mask': None, 'V_mask': None}
        trs_embedding, bert_embedding = self.bmt([trs_embedding, bert_embedding], mask)
        fustion_embedding = torch.cat((trs_embedding, bert_embedding), dim=-1)
        embedding = self.bridge(fustion_embedding)
        embedding = torch.mean(embedding, dim=1)
        return embedding


class MyBMTHiddenFustionModel(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.bmt = BiModalEncoder(d_model_A=1536,
                                  d_model_V=768,
                                  d_model=None,
                                  dout_p=0.1,
                                  H=8,
                                  d_ff_A=3072,
                                  d_ff_V=1536,
                                  N=2)
        self.bridge = BridgeConnection(2304, 256, 0.1)

    def forward(self, trs_hidden, bert_hidden):
        mask = {'A_mask': None, 'V_mask': None}
        trs_embedding, bert_embedding = self.bmt([trs_hidden, bert_hidden], mask)
        fustion_embedding = torch.cat((trs_embedding, bert_embedding), dim=-1)
        # embedding_max = torch.max(fustion_embedding, dim=1)[0]
        embedding_mean = torch.mean(fustion_embedding, dim=1)
        embedding = self.bridge(embedding_mean)

        return embedding


class MyConcatSeFustionModel(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.trs_linear = nn.Linear(cfg['TRS_INPUT'], cfg['OUPUT_SIZE'])
        self.bert_linear = nn.Linear(cfg['BERT_INPUT'], cfg['OUPUT_SIZE'])
        self.fusion_linear = nn.Linear(cfg['TRS_INPUT'] + cfg['BERT_INPUT'], cfg['OUPUT_SIZE'])
        # self.trs_se = nn.Sequential(nn.Linear(cfg['OUPUT_SIZE'],
        #                                       cfg['OUPUT_SIZE'] // 16), nn.ReLU(),
        #                             nn.Linear(cfg['OUPUT_SIZE'] // 16, cfg['OUPUT_SIZE']),
        #                             nn.Sigmoid())
        # self.bert_se = nn.Sequential(nn.Linear(cfg['OUPUT_SIZE'],
        #                                        cfg['OUPUT_SIZE'] // 16), nn.ReLU(),
        #                              nn.Linear(cfg['OUPUT_SIZE'] // 16, cfg['OUPUT_SIZE']),
        #                              nn.Sigmoid())
        self.trs_se = nn.Sequential(nn.Linear(cfg['OUPUT_SIZE'], cfg['OUPUT_SIZE']),
                                    nn.BatchNorm1d(cfg['OUPUT_SIZE']), nn.Sigmoid())
        self.bert_se = nn.Sequential(nn.Linear(cfg['OUPUT_SIZE'], cfg['OUPUT_SIZE']),
                                     nn.BatchNorm1d(cfg['OUPUT_SIZE']), nn.Sigmoid())
        self.fusion_se = nn.Sequential(nn.Linear(cfg['OUPUT_SIZE'], cfg['OUPUT_SIZE']),
                                       nn.BatchNorm1d(cfg['OUPUT_SIZE']), nn.Sigmoid())
        self.drop_out = nn.Dropout(p=cfg['DROPOUT'])

    def forward(self, trs_embedding, bert_embedding):
        trs_embedding = self.drop_out(trs_embedding)
        bert_embedding = self.drop_out(bert_embedding)
        fusion_embedding = torch.cat((trs_embedding, bert_embedding), dim=-1)

        trs_embedding = self.trs_linear(trs_embedding)
        bert_embedding = self.bert_linear(bert_embedding)
        fusion_embedding = self.fusion_linear(fusion_embedding)

        trs_embedding_w = self.trs_se(trs_embedding)
        bert_embedding_w = self.bert_se(bert_embedding)
        fusion_embedding_w = self.fusion_se(fusion_embedding)

        trs_embedding = torch.nn.functional.normalize(trs_embedding, p=2, dim=1)
        bert_embedding = torch.nn.functional.normalize(bert_embedding, p=2, dim=1)
        fusion_embedding = torch.nn.functional.normalize(fusion_embedding, p=2, dim=1)

        embedding = trs_embedding * trs_embedding_w + bert_embedding * bert_embedding_w + fusion_embedding * fusion_embedding_w
        # embedding = trs_embedding * trs_embedding_w + bert_embedding * bert_embedding_w
        return embedding


class MyConcatSeFustionModel2(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        concat_len = cfg['TRS_INPUT'] + cfg['BERT_INPUT']
        self.fusion_linear = nn.Linear(concat_len, cfg['OUPUT_SIZE'])
        self.fusion_se = nn.Sequential(nn.Linear(concat_len, concat_len // 16, bias=False),
                                       nn.ReLU(), nn.Linear(concat_len // 16,
                                                            concat_len,
                                                            bias=False), nn.Sigmoid())
        self.drop_out = nn.Dropout(p=cfg['DROPOUT'])

    def forward(self, trs_embedding, bert_embedding):
        fusion_embedding = self.drop_out(torch.cat((trs_embedding, bert_embedding), dim=-1))
        fusion_embedding = self.fusion_se(fusion_embedding) * fusion_embedding
        embedding = self.fusion_linear(fusion_embedding)
        return embedding


class MyConcatResidualSeFustionModel(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        concat_len = cfg['TRS_INPUT'] + cfg['BERT_INPUT']
        self.fusion_linear = nn.Linear(concat_len, cfg['OUPUT_SIZE'])
        self.fusion_se = nn.Sequential(nn.Linear(concat_len, concat_len // 16, bias=False),
                                       nn.ReLU(), nn.Linear(concat_len // 16,
                                                            concat_len,
                                                            bias=False), nn.Sigmoid())
        self.drop_out = nn.Dropout(p=cfg['DROPOUT'])

    def forward(self, trs_embedding, bert_embedding):
        fusion_embedding = self.drop_out(torch.cat((trs_embedding, bert_embedding), dim=-1))
        fusion_embedding = self.fusion_se(fusion_embedding) * fusion_embedding + fusion_embedding
        embedding = self.fusion_linear(fusion_embedding)
        return embedding


class MySeFustionDropModel(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.trs_linear = nn.Linear(cfg['TRS_INPUT'], cfg['OUPUT_SIZE'])
        self.bert_linear = nn.Linear(cfg['BERT_INPUT'], cfg['OUPUT_SIZE'])
        self.trs_se = nn.Sequential(nn.Linear(cfg['OUPUT_SIZE'], cfg['OUPUT_SIZE']),
                                    nn.BatchNorm1d(cfg['OUPUT_SIZE']), nn.Sigmoid())
        self.bert_se = nn.Sequential(nn.Linear(cfg['OUPUT_SIZE'], cfg['OUPUT_SIZE']),
                                     nn.BatchNorm1d(cfg['OUPUT_SIZE']), nn.Sigmoid())
        self.drop_out = nn.Dropout(p=cfg['DROPOUT'])

    def forward(self, trs_embedding, bert_embedding):
        trs_embedding = self.drop_out(trs_embedding)
        bert_embedding = self.drop_out(bert_embedding)

        trs_embedding = self.trs_linear(trs_embedding)
        bert_embedding = self.bert_linear(bert_embedding)

        trs_embedding_w = self.trs_se(trs_embedding)
        bert_embedding_w = self.bert_se(bert_embedding)

        trs_embedding = torch.nn.functional.normalize(trs_embedding, p=2, dim=1)
        bert_embedding = torch.nn.functional.normalize(bert_embedding, p=2, dim=1)

        embedding = trs_embedding * trs_embedding_w + bert_embedding * bert_embedding_w
        return embedding


class MySeFustionAllModel(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.trs_linear = nn.Linear(cfg['TRS_INPUT'], cfg['OUPUT_SIZE'])
        self.bert_linear = nn.Linear(cfg['BERT_INPUT'], cfg['OUPUT_SIZE'])
        self.trs_se = nn.Sequential(
            nn.Linear(cfg['TRS_INPUT'] + cfg['BERT_INPUT'], cfg['OUPUT_SIZE']),
            nn.BatchNorm1d(cfg['OUPUT_SIZE']), nn.Sigmoid())
        self.bert_se = nn.Sequential(
            nn.Linear(cfg['TRS_INPUT'] + cfg['BERT_INPUT'], cfg['OUPUT_SIZE']),
            nn.BatchNorm1d(cfg['OUPUT_SIZE']), nn.Sigmoid())
        self.drop_out = nn.Dropout(p=cfg['DROPOUT'])

    def forward(self, trs_embedding, bert_embedding):
        # trs_embedding = self.drop_out(trs_embedding)
        # bert_embedding = self.drop_out(bert_embedding)

        features = torch.cat((trs_embedding, bert_embedding), dim=1)
        trs_embedding = self.trs_linear(trs_embedding)
        bert_embedding = self.bert_linear(bert_embedding)

        trs_embedding_w = self.trs_se(features)
        bert_embedding_w = self.bert_se(features)

        trs_embedding = torch.nn.functional.normalize(trs_embedding, p=2, dim=1)
        bert_embedding = torch.nn.functional.normalize(bert_embedding, p=2, dim=1)

        embedding = trs_embedding * trs_embedding_w + bert_embedding * bert_embedding_w
        return embedding


class MyHiddenFustionModel(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)

        self.norm = nn.BatchNorm1d((cfg['TRS_INPUT'] + cfg['BERT_INPUT']) * 2)
        self.dense = nn.Linear((cfg['TRS_INPUT'] + cfg['BERT_INPUT']) * 2, 512)
        self.norm_1 = nn.BatchNorm1d(512)
        self.dense_1 = nn.Linear(512, 256)
        self.dropout = nn.Dropout(0.2)

    def forward(self, trs_hidden, bert_hidden):
        trs_hidden_max = torch.max(trs_hidden, dim=1)[0]
        trs_hidden_mean = torch.mean(trs_hidden, dim=1)
        bert_hidden_max = torch.max(bert_hidden, dim=1)[0]
        bert_hidden_mean = torch.mean(bert_hidden, dim=1)

        x = torch.cat((trs_hidden_max, trs_hidden_mean, bert_hidden_max, bert_hidden_mean), dim=-1)

        x = self.norm(x)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(self.norm_1(x))
        x = self.dropout(x)
        x = self.dense_1(x)
        return x


class MySeThreeFustionModel(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.tag_trs_linear = nn.Linear(cfg['TRS_INPUT'], cfg['OUPUT_SIZE'])
        self.tag_bert_linear = nn.Linear(cfg['BERT_INPUT'], cfg['OUPUT_SIZE'])
        self.cat_trs_linear = nn.Linear(cfg['TRS_INPUT'], cfg['OUPUT_SIZE'])

        self.tag_trs_se = nn.Sequential(nn.Linear(cfg['OUPUT_SIZE'], cfg['OUPUT_SIZE']),
                                        nn.BatchNorm1d(cfg['OUPUT_SIZE']), nn.Sigmoid())
        self.tag_bert_se = nn.Sequential(nn.Linear(cfg['OUPUT_SIZE'], cfg['OUPUT_SIZE']),
                                         nn.BatchNorm1d(cfg['OUPUT_SIZE']), nn.Sigmoid())
        self.cat_trs_se = nn.Sequential(nn.Linear(cfg['OUPUT_SIZE'], cfg['OUPUT_SIZE']),
                                        nn.BatchNorm1d(cfg['OUPUT_SIZE']), nn.Sigmoid())
        self.drop_out = nn.Dropout(p=cfg['DROPOUT'])

    def forward(self, tag_trs_embedding, tag_bert_embedding, cat_trs_embedding):

        tag_trs_embedding = self.tag_trs_linear(self.drop_out(tag_trs_embedding))
        tag_bert_embedding = self.tag_bert_linear(self.drop_out(tag_bert_embedding))
        cat_trs_embedding = self.cat_trs_linear(self.drop_out(cat_trs_embedding))

        tag_trs_embedding_w = self.tag_trs_se(tag_trs_embedding)
        tag_bert_embedding_w = self.tag_bert_se(tag_bert_embedding)
        cat_trs_embedding_w = self.cat_trs_se(cat_trs_embedding)

        tag_trs_embedding = torch.nn.functional.normalize(tag_trs_embedding, p=2, dim=1)
        tag_bert_embedding = torch.nn.functional.normalize(tag_bert_embedding, p=2, dim=1)
        cat_trs_embedding = torch.nn.functional.normalize(cat_trs_embedding, p=2, dim=1)

        embedding = (tag_trs_embedding * tag_trs_embedding_w +
                     tag_bert_embedding * tag_bert_embedding_w +
                     cat_trs_embedding * cat_trs_embedding_w)
        return embedding


class MySeFourFustionModel(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.tag_trs_linear = nn.Linear(cfg['TRS_INPUT'], cfg['OUPUT_SIZE'])
        self.tag_bert_linear = nn.Linear(cfg['BERT_INPUT'], cfg['OUPUT_SIZE'])
        self.cat_trs_linear = nn.Linear(cfg['TRS_INPUT'], cfg['OUPUT_SIZE'])
        self.cat_bert_linear = nn.Linear(cfg['BERT_INPUT'], cfg['OUPUT_SIZE'])

        self.tag_trs_se = nn.Sequential(nn.Linear(cfg['OUPUT_SIZE'], cfg['OUPUT_SIZE']),
                                        nn.BatchNorm1d(cfg['OUPUT_SIZE']), nn.Sigmoid())
        self.tag_bert_se = nn.Sequential(nn.Linear(cfg['OUPUT_SIZE'], cfg['OUPUT_SIZE']),
                                         nn.BatchNorm1d(cfg['OUPUT_SIZE']), nn.Sigmoid())
        self.cat_trs_se = nn.Sequential(nn.Linear(cfg['OUPUT_SIZE'], cfg['OUPUT_SIZE']),
                                        nn.BatchNorm1d(cfg['OUPUT_SIZE']), nn.Sigmoid())
        self.cat_bert_se = nn.Sequential(nn.Linear(cfg['OUPUT_SIZE'], cfg['OUPUT_SIZE']),
                                         nn.BatchNorm1d(cfg['OUPUT_SIZE']), nn.Sigmoid())
        self.drop_out = nn.Dropout(p=cfg['DROPOUT'])

    def forward(self, tag_trs_embedding, tag_bert_embedding, cat_trs_embedding, cat_bert_embedding):

        tag_trs_embedding = self.tag_trs_linear(self.drop_out(tag_trs_embedding))
        tag_bert_embedding = self.tag_bert_linear(self.drop_out(tag_bert_embedding))
        cat_trs_embedding = self.cat_trs_linear(self.drop_out(cat_trs_embedding))
        cat_bert_embedding = self.cat_bert_linear(self.drop_out(cat_bert_embedding))

        tag_trs_embedding_w = self.tag_trs_se(tag_trs_embedding)
        tag_bert_embedding_w = self.tag_bert_se(tag_bert_embedding)
        cat_trs_embedding_w = self.cat_trs_se(cat_trs_embedding)
        cat_bert_embedding_w = self.cat_bert_se(cat_bert_embedding)

        tag_trs_embedding = torch.nn.functional.normalize(tag_trs_embedding, p=2, dim=1)
        tag_bert_embedding = torch.nn.functional.normalize(tag_bert_embedding, p=2, dim=1)
        cat_trs_embedding = torch.nn.functional.normalize(cat_trs_embedding, p=2, dim=1)
        cat_bert_embedding = torch.nn.functional.normalize(cat_bert_embedding, p=2, dim=1)

        embedding = (tag_trs_embedding * tag_trs_embedding_w +
                     tag_bert_embedding * tag_bert_embedding_w +
                     cat_trs_embedding * cat_trs_embedding_w +
                     cat_bert_embedding * cat_bert_embedding_w)
        return embedding


class MySeFourFustionMeanModel(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.tag_trs_linear = nn.Linear(cfg['TRS_INPUT'], cfg['OUPUT_SIZE'])
        self.tag_bert_linear = nn.Linear(cfg['BERT_INPUT'], cfg['OUPUT_SIZE'])
        self.cat_trs_linear = nn.Linear(cfg['TRS_INPUT'], cfg['OUPUT_SIZE'])
        self.cat_bert_linear = nn.Linear(cfg['BERT_INPUT'], cfg['OUPUT_SIZE'])

        self.tag_trs_se = nn.Sequential(nn.Linear(cfg['OUPUT_SIZE'], cfg['OUPUT_SIZE']),
                                        nn.BatchNorm1d(cfg['OUPUT_SIZE']), nn.Sigmoid())
        self.tag_bert_se = nn.Sequential(nn.Linear(cfg['OUPUT_SIZE'], cfg['OUPUT_SIZE']),
                                         nn.BatchNorm1d(cfg['OUPUT_SIZE']), nn.Sigmoid())
        self.cat_trs_se = nn.Sequential(nn.Linear(cfg['OUPUT_SIZE'], cfg['OUPUT_SIZE']),
                                        nn.BatchNorm1d(cfg['OUPUT_SIZE']), nn.Sigmoid())
        self.cat_bert_se = nn.Sequential(nn.Linear(cfg['OUPUT_SIZE'], cfg['OUPUT_SIZE']),
                                         nn.BatchNorm1d(cfg['OUPUT_SIZE']), nn.Sigmoid())
        self.drop_out = nn.Dropout(p=cfg['DROPOUT'])
        # self.weight = nn.Parameter(torch.tensor([1, 1, 0.5, 0.3]))

    def forward(self, tag_trs_embedding, tag_bert_embedding, cat_trs_embedding, cat_bert_embedding):

        tag_trs_embedding = self.tag_trs_linear(self.drop_out(tag_trs_embedding))
        tag_bert_embedding = self.tag_bert_linear(self.drop_out(tag_bert_embedding))
        cat_trs_embedding = self.cat_trs_linear(self.drop_out(cat_trs_embedding))
        cat_bert_embedding = self.cat_bert_linear(self.drop_out(cat_bert_embedding))

        tag_trs_embedding_w = self.tag_trs_se(tag_trs_embedding)
        tag_bert_embedding_w = self.tag_bert_se(tag_bert_embedding)
        cat_trs_embedding_w = self.cat_trs_se(cat_trs_embedding)
        cat_bert_embedding_w = self.cat_bert_se(cat_bert_embedding)

        tag_trs_embedding = torch.nn.functional.normalize(tag_trs_embedding, p=2, dim=1)
        tag_bert_embedding = torch.nn.functional.normalize(tag_bert_embedding, p=2, dim=1)
        cat_trs_embedding = torch.nn.functional.normalize(cat_trs_embedding, p=2, dim=1)
        cat_bert_embedding = torch.nn.functional.normalize(cat_bert_embedding, p=2, dim=1)

        embedding = (
            tag_trs_embedding * tag_trs_embedding_w + tag_bert_embedding * tag_bert_embedding_w +
            cat_trs_embedding * cat_trs_embedding_w + cat_bert_embedding * cat_bert_embedding_w) / 4
        return embedding


class MyConcatWeightModel(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.tag_trs_linear = nn.Linear(cfg['TRS_INPUT'], cfg['OUPUT_SIZE'])
        self.tag_bert_linear = nn.Linear(cfg['BERT_INPUT'], cfg['OUPUT_SIZE'])
        self.cat_trs_linear = nn.Linear(cfg['TRS_INPUT'], cfg['OUPUT_SIZE'])
        self.cat_bert_linear = nn.Linear(cfg['BERT_INPUT'], cfg['OUPUT_SIZE'])
        self.concat_linear = nn.Linear((cfg['TRS_INPUT'] + cfg['BERT_INPUT']) * 2,
                                       cfg['OUPUT_SIZE'])

        self.drop_out = nn.Dropout(p=cfg['DROPOUT'])
        weight = torch.ones((5, cfg['OUPUT_SIZE'])) * torch.tensor([1, 1, 1, 1, 1]).reshape(5, 1)
        self.weight = nn.Parameter(weight)

    def forward(self, tag_trs_embedding, tag_bert_embedding, cat_trs_embedding, cat_bert_embedding):

        tag_trs_embedding = self.drop_out(tag_trs_embedding)
        tag_bert_embedding = self.drop_out(tag_bert_embedding)
        cat_trs_embedding = self.drop_out(cat_trs_embedding)
        cat_bert_embedding = self.drop_out(cat_bert_embedding)

        concat_embedding = torch.cat(
            (tag_trs_embedding, tag_bert_embedding, cat_trs_embedding, cat_bert_embedding), dim=1)

        tag_trs_embedding = self.tag_trs_linear(tag_trs_embedding)
        tag_bert_embedding = self.tag_bert_linear(tag_bert_embedding)
        cat_trs_embedding = self.cat_trs_linear(cat_trs_embedding)
        cat_bert_embedding = self.cat_bert_linear(cat_bert_embedding)
        concat_embedding = self.concat_linear(concat_embedding)

        tag_trs_embedding = torch.nn.functional.normalize(tag_trs_embedding, p=2, dim=1)
        tag_bert_embedding = torch.nn.functional.normalize(tag_bert_embedding, p=2, dim=1)
        cat_trs_embedding = torch.nn.functional.normalize(cat_trs_embedding, p=2, dim=1)
        cat_bert_embedding = torch.nn.functional.normalize(cat_bert_embedding, p=2, dim=1)
        concat_embedding = torch.nn.functional.normalize(concat_embedding, p=2, dim=1)

        fusdion_embedding = torch.stack((tag_trs_embedding, tag_bert_embedding, cat_trs_embedding,
                                         cat_bert_embedding, concat_embedding),
                                        dim=1)

        fusdion_embedding = fusdion_embedding * self.weight
        embedding = torch.sum(fusdion_embedding, dim=-2)

        return embedding


class MyConcatSeModel(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        concat_size = (cfg['TRS_INPUT'] + cfg['BERT_INPUT']) * 2
        self.concat_se = nn.Sequential(nn.Linear(concat_size, concat_size),
                                       nn.BatchNorm1d(concat_size), nn.Sigmoid())
        self.concat_linear = nn.Linear(concat_size, cfg['OUPUT_SIZE'])

    def forward(self, tag_trs_embedding, tag_bert_embedding, cat_trs_embedding, cat_bert_embedding):

        concat_embedding = torch.cat(
            (tag_trs_embedding, tag_bert_embedding, cat_trs_embedding, cat_bert_embedding), dim=1)

        concat_embedding = concat_embedding * self.concat_se(concat_embedding)

        embedding = self.concat_linear(concat_embedding)

        return embedding


class MyFinetuneModel0(TaskBase):
    def __init__(self, cfg, tag_nextvlad, tag_bert):
        super().__init__(cfg)
        self.tag_nextvlad = tag_nextvlad
        self.tag_bert = tag_bert
        self.fustion = MySeFustionAllModel(cfg['MySeNextFustionModel'])
        self.model_list = [self.fustion]

    def forward(self, frame_feature, asr_title):
        frame_feature = frame_feature.cuda()
        nextvlad_embedding = self.tag_nextvlad(frame_feature)
        _, bert_embedding = self.tag_bert(asr_title)

        embedding = self.fustion(nextvlad_embedding, bert_embedding)
        normed_embedding = nn.functional.normalize(embedding, p=2, dim=1)

        b, n = normed_embedding.shape
        embedding_1, embedding_2 = normed_embedding[:b // 2], normed_embedding[b // 2:]

        cos = torch.mul(embedding_1, embedding_2)
        cos = torch.sum(cos, dim=1)

        return cos, normed_embedding


# 两个模型1w类训练


class MyFinetuneRetrainModel(TaskBase):
    def __init__(self, cfg, tag_trs, tag_bert, fusion):
        super().__init__(cfg)
        self.tag_trs = tag_trs
        self.tag_bert = tag_bert
        self.fusion = fusion
        self.model_list = [self.tag_trs, self.tag_bert, self.fusion]

    def forward(self, frame_feature, mask, asr_title):
        frame_feature = frame_feature.cuda()
        mask = mask.cuda()

        trs_hidden, _ = self.tag_trs(frame_feature, mask)
        bert_hidden, _ = self.tag_bert(asr_title)

        embedding = self.fusion(trs_hidden[:, 1:, :], bert_hidden[:, :, :])
        normed_embedding = nn.functional.normalize(embedding, p=2, dim=1)

        b, n = normed_embedding.shape
        embedding_1, embedding_2 = normed_embedding[:b // 2], normed_embedding[b // 2:]

        cos = torch.mul(embedding_1, embedding_2)
        cos = torch.sum(cos, dim=1)

        return cos, normed_embedding


class MyRetrainModel(TaskBase):
    def __init__(self, cfg, tag_trs, tag_bert):
        super().__init__(cfg)
        self.tag_trs = tag_trs
        self.tag_bert = tag_bert
        self.fustion = MyHiddenFustionModel(cfg['MySeFustionModel'])
        self.class_head = nn.Sequential(nn.Linear(256, 10000))
        self.model_list = [self.fustion]

    def forward(self, frame_feature, mask, asr_title):
        frame_feature = frame_feature.cuda()
        mask = mask.cuda()
        trs_hidden, _ = self.tag_trs(frame_feature, mask)
        bert_hidden, _ = self.tag_bert(asr_title)

        x = self.fustion(trs_hidden[:, 1:, :], bert_hidden[:, :, :])
        normed_embedding = nn.functional.normalize(x, p=2, dim=1)
        pred = torch.sigmoid(self.class_head(x))

        return pred, normed_embedding


class MyRetrainMode2(TaskBase):
    def __init__(self, cfg, tag_trs, tag_bert):
        super().__init__(cfg)
        self.tag_trs = tag_trs
        self.tag_bert = tag_bert
        self.fustion = MyBMTHiddenFustionModel(cfg['MySeFustionModel'])
        self.class_head = nn.Sequential(nn.Linear(256, 10000))
        self.model_list = [self.fustion]

    def forward(self, frame_feature, mask, asr_title):
        frame_feature = frame_feature.cuda()
        mask = mask.cuda()
        trs_hidden, _ = self.tag_trs(frame_feature, mask)
        bert_hidden, _ = self.tag_bert(asr_title)

        x = self.fustion(trs_hidden[:, 1:33, :], bert_hidden[:, 0:32, :])
        normed_embedding = nn.functional.normalize(x, p=2, dim=1)
        pred = torch.sigmoid(self.class_head(x))

        return pred, normed_embedding


# 单模型finetune
class MySingleTrsFinetuneModel(TaskBase):
    def __init__(self, cfg, trs):
        super().__init__(cfg)
        self.trs = trs
        self.model_list = [self.model]

    def forward(self, frame_feature, mask):
        frame_feature = frame_feature.cuda()
        mask = mask.cuda()
        _, embedding = self.trs(frame_feature, mask)

        normed_embedding = nn.functional.normalize(embedding, p=2, dim=1)

        b, n = normed_embedding.shape
        embedding_1, embedding_2 = normed_embedding[:b // 2], normed_embedding[b // 2:]

        cos = torch.mul(embedding_1, embedding_2)
        cos = torch.sum(cos, dim=1)

        return cos, normed_embedding


class MySingleBertFinetuneModel(TaskBase):
    def __init__(self, cfg, bert):
        super().__init__(cfg)
        self.bert = bert
        self.model_list = [self.model]

    def forward(self, title_asr):
        _, embedding = self.bert(title_asr)

        normed_embedding = nn.functional.normalize(embedding, p=2, dim=1)

        b, n = normed_embedding.shape
        embedding_1, embedding_2 = normed_embedding[:b // 2], normed_embedding[b // 2:]

        cos = torch.mul(embedding_1, embedding_2)
        cos = torch.sum(cos, dim=1)

        return cos, normed_embedding


#  两个模型融合
class MyFinetuneModel1(TaskBase):
    def __init__(self, cfg, tag_trs, tag_bert):
        super().__init__(cfg)
        self.tag_trs = tag_trs
        self.tag_bert = tag_bert
        self.fustion = MySeFustionAllModel(cfg['MySeFustionModel'])
        self.model_list = [self.fustion]

    def forward(self, frame_feature, mask, asr_title):
        frame_feature = frame_feature.cuda()
        mask = mask.cuda()
        _, trs_embedding = self.tag_trs(frame_feature, mask)
        _, bert_embedding = self.tag_bert(asr_title)

        embedding = self.fustion(trs_embedding, bert_embedding)
        normed_embedding = nn.functional.normalize(embedding, p=2, dim=1)

        b, n = normed_embedding.shape
        embedding_1, embedding_2 = normed_embedding[:b // 2], normed_embedding[b // 2:]

        cos = torch.mul(embedding_1, embedding_2)
        cos = torch.sum(cos, dim=1)

        return cos, normed_embedding


class MyReFinetuneModel(TaskBase):
    def __init__(self, cfg, tag_trs, tag_bert, fustion):
        super().__init__(cfg)
        self.tag_trs = tag_trs
        self.tag_bert = tag_bert
        self.fustion = fustion
        self.model_list = [self.fustion]

    def forward(self, frame_feature, mask, asr_title):
        frame_feature = frame_feature.cuda()
        mask = mask.cuda()
        _, trs_embedding = self.tag_trs(frame_feature, mask)
        _, bert_embedding = self.tag_bert(asr_title)

        embedding = self.fustion(trs_embedding, bert_embedding)
        normed_embedding = nn.functional.normalize(embedding, p=2, dim=1)

        b, n = normed_embedding.shape
        embedding_1, embedding_2 = normed_embedding[:b // 2], normed_embedding[b // 2:]

        cos = torch.mul(embedding_1, embedding_2)
        cos = torch.sum(cos, dim=1)

        return cos, normed_embedding


# max mean pooling
class MyFinetuneModel2(TaskBase):
    def __init__(self, cfg, tag_trs, tag_bert):
        super().__init__(cfg)
        self.tag_trs = tag_trs
        self.tag_bert = tag_bert
        self.fustion = MyHiddenFustionModel(cfg['MySeFustionModel'])
        self.model_list = [self.fustion]

    def forward(self, frame_feature, mask, asr_title):
        frame_feature = frame_feature.cuda()
        mask = mask.cuda()
        trs_hidden, _ = self.tag_trs(frame_feature, mask)
        bert_hidden, _ = self.tag_bert(asr_title)

        embedding = self.fustion(trs_hidden[:, 1:, :], bert_hidden[:, :, :])
        normed_embedding = nn.functional.normalize(embedding, p=2, dim=1)

        b, n = normed_embedding.shape
        embedding_1, embedding_2 = normed_embedding[:b // 2], normed_embedding[b // 2:]

        cos = torch.mul(embedding_1, embedding_2)
        cos = torch.sum(cos, dim=1)

        return cos, normed_embedding


#  三个模型融合
class MyFinetuneModel3(TaskBase):
    def __init__(self, cfg, tag_trs, tag_bert, cat_trs):
        super().__init__(cfg)
        self.tag_trs = tag_trs
        self.tag_bert = tag_bert
        self.cat_trs = cat_trs
        self.fustion = MySeThreeFustionModel(cfg['MyFourFustionModel'])
        self.model_list = [self.fustion]

    def forward(self, frame_feature, mask, asr_title):
        frame_feature = frame_feature.cuda()
        mask = mask.cuda()
        tag_trs_embedding = self.tag_trs(frame_feature, mask)
        _, tag_bert_embedding = self.tag_bert(asr_title)
        cat_trs_embedding = self.cat_trs(frame_feature, mask)

        embedding = self.fustion(tag_trs_embedding, tag_bert_embedding, cat_trs_embedding)
        normed_embedding = nn.functional.normalize(embedding, p=2, dim=1)

        b, n = normed_embedding.shape
        embedding_1, embedding_2 = normed_embedding[:b // 2], normed_embedding[b // 2:]

        cos = torch.mul(embedding_1, embedding_2)
        cos = torch.sum(cos, dim=1)

        return cos, normed_embedding


#  四个模型融合
class MyFinetuneModel4(TaskBase):
    def __init__(self, cfg, tag_trs, tag_bert, cat_trs, cat_bert):
        super().__init__(cfg)
        self.tag_trs = tag_trs
        self.tag_bert = tag_bert
        self.cat_trs = cat_trs
        self.cat_bert = cat_bert
        self.fustion = MySeFourFustionMeanModel(cfg['MyFourFustionModel'])
        self.model_list = [self.fustion]

    def forward(self, frame_feature, mask, asr_title):
        frame_feature = frame_feature.cuda()
        mask = mask.cuda()
        tag_trs_embedding = self.tag_trs(frame_feature, mask)
        _, tag_bert_embedding = self.tag_bert(asr_title)
        cat_trs_embedding = self.cat_trs(frame_feature, mask)
        _, cat_bert_embedding = self.cat_bert(asr_title)

        embedding = self.fustion(tag_trs_embedding, tag_bert_embedding, cat_trs_embedding,
                                 cat_bert_embedding)
        normed_embedding = nn.functional.normalize(embedding, p=2, dim=1)

        b, n = normed_embedding.shape
        embedding_1, embedding_2 = normed_embedding[:b // 2], normed_embedding[b // 2:]

        cos = torch.mul(embedding_1, embedding_2)
        cos = torch.sum(cos, dim=1)

        return cos, normed_embedding


class MyFinetuneMLM(TaskBase):
    def __init__(self, cfg, mlm, trs):
        super().__init__(cfg)
        self.mlm = mlm
        self.trs = trs
        self.fustion = MySeFustionAllModel(cfg['SE_FUSTION'])

    def forward(self,
                frame_feature=None,
                frame_masks=None,
                input_ids=None,
                token_type_ids=None,
                text_mask=None):

        # 768
        _, text_embedding = self.mlm(frame_feature=frame_feature,
                                     frame_masks=frame_masks,
                                     input_ids=input_ids,
                                     text_mask=text_mask,
                                     token_type_ids=token_type_ids)
        # 1536
        _, video_embedding = self.trs(frame_feature, frame_masks)

        # 取class token
        text_embedding = text_embedding[:, 0]

        embedding = self.fustion(video_embedding, text_embedding)

        normed_embedding = nn.functional.normalize(embedding, p=2, dim=1)

        b, n = normed_embedding.shape
        embedding_1, embedding_2 = normed_embedding[:b // 2], normed_embedding[b // 2:]

        cos = torch.mul(embedding_1, embedding_2)
        cos = torch.sum(cos, dim=1)

        return cos, normed_embedding
