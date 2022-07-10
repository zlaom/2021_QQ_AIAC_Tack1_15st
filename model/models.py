import torch
import torch.nn as nn
from model.model_base import ModelBase, TaskBase
from model.my_bert import MacBert
from model.nextvlad import NeXtVLAD
from model.my_transformers import FrameFeatureTrs
from model.finetune_model import MySeFustionAllModel


class MyFrameFeatureTrs(ModelBase):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.trs = FrameFeatureTrs(max_frame=cfg['MAX_FRAME'],
                                   frame_dim=cfg['FRAME_DIM'],
                                   dim=cfg['DIM'],
                                   depth=cfg['DEPTH'],
                                   heads=cfg['HEADS'],
                                   mlp_dim=cfg['MLP_DIM'],
                                   dim_head=cfg['DIM_HEAD'],
                                   pool=cfg['POOL'],
                                   pos_emb=cfg['POS_EMB'],
                                   dropout=cfg['DROPOUT'],
                                   emb_dropout=cfg['EMB_DROPOUT'])

    def forward(self, farme_featrue, mask):
        embedding = self.trs(farme_featrue, mask)
        return embedding


class TitleVideoConcatSeClsA(ModelBase):
    def __init__(self, cfg):
        """
        :param cfg: concat_cls config defined in your_config.yaml
        """
        super().__init__(cfg)
        self.video_linear = nn.Linear(cfg['VIDEO_INPUT_SIZE'], cfg['HIDDEN_SIZE'])
        self.title_linear = nn.Linear(cfg['TITLE_INPUT_SIZE'], cfg['HIDDEN_SIZE'])
        self.concat_linear = nn.Linear(cfg['TITLE_INPUT_SIZE'] + cfg['VIDEO_INPUT_SIZE'],
                                       cfg['HIDDEN_SIZE'])
        self.fc_logits = nn.Linear(cfg['HIDDEN_SIZE'], cfg['NUM_CLASSES'])
        se_weight = torch.ones((3, cfg['HIDDEN_SIZE'])) * torch.tensor([1, 1, 1]).reshape(3, 1)
        self.se_weight = nn.Parameter(se_weight)
        self.title_dropout = nn.Dropout(p=cfg['TITLE_DROPOUT'])
        self.video_dropout = nn.Dropout(p=cfg['VIDEO_DROPOUT'])
        self.fusion_dropout = nn.Dropout(p=cfg['VIDEO_DROPOUT'])

    def forward(self, video_feature, title_feature):
        """
        :param title_feature: title feature extracted from title representation
        :param label: classification target
        :return: (predictions, embeddings), model loss
        """
        concat_feature = torch.cat((video_feature, title_feature), dim=1)

        title_feature = self.title_dropout(title_feature)
        video_feature = self.video_dropout(video_feature)
        # concat_feature = torch.cat((video_feature, title_feature), dim=1)
        concat_feature = self.fusion_dropout(concat_feature)

        video_hidden = self.video_linear(video_feature)
        title_hidden = self.title_linear(title_feature)
        concat_hidden = self.concat_linear(concat_feature)

        concat_hidden = torch.stack((video_hidden, title_hidden, concat_hidden), dim=1)
        concat_hidden = concat_hidden * self.se_weight
        embedding = torch.sum(concat_hidden, dim=-2)

        normed_embedding = nn.functional.normalize(embedding, p=2, dim=1)

        pred = self.fc_logits(torch.relu(embedding))
        pred = torch.sigmoid(pred)
        return pred, normed_embedding


# Model
class TitleVideoConcatSeCls(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.video_linear = nn.Linear(cfg['VIDEO_INPUT_SIZE'], cfg['HIDDEN_SIZE'])
        self.title_linear = nn.Linear(cfg['TITLE_INPUT_SIZE'], cfg['HIDDEN_SIZE'])
        self.concat_linear = nn.Linear(cfg['TITLE_INPUT_SIZE'] + cfg['VIDEO_INPUT_SIZE'],
                                       cfg['HIDDEN_SIZE'])
        self.fc_logits = nn.Linear(cfg['HIDDEN_SIZE'], cfg['NUM_CLASSES'])
        se_weight = torch.ones((3, cfg['HIDDEN_SIZE'])) * torch.tensor([1, 1, 1]).reshape(3, 1)
        self.se_weight = nn.Parameter(se_weight)
        self.title_dropout = nn.Dropout(p=cfg['TITLE_DROPOUT'])
        self.video_dropout = nn.Dropout(p=cfg['VIDEO_DROPOUT'])

    def forward(self, video_feature, title_feature):
        title_feature = self.title_dropout(title_feature)
        video_feature = self.video_dropout(video_feature)
        concat_feature = torch.cat((video_feature, title_feature), dim=1)

        video_hidden = self.video_linear(video_feature)
        title_hidden = self.title_linear(title_feature)
        concat_hidden = self.concat_linear(concat_feature)

        concat_hidden = torch.stack((video_hidden, title_hidden, concat_hidden), dim=1)
        concat_hidden = concat_hidden * self.se_weight
        embedding = torch.sum(concat_hidden, dim=-2)

        normed_embedding = nn.functional.normalize(embedding, p=2, dim=1)

        pred = self.fc_logits(torch.relu(embedding))
        pred = torch.sigmoid(pred)
        return pred, normed_embedding


class TitleVideoConcatSeCls2(ModelBase):
    def __init__(self, cfg):
        """
        :param cfg: concat_cls config defined in your_config.yaml
        """
        super().__init__(cfg)
        # self.video_linear = nn.Linear(cfg['VIDEO_INPUT_SIZE'], cfg['HIDDEN_SIZE'])
        # self.title_linear = nn.Linear(cfg['TITLE_INPUT_SIZE'], cfg['HIDDEN_SIZE'])
        concat_size = cfg['TITLE_INPUT_SIZE'] + cfg['VIDEO_INPUT_SIZE']
        self.concat_linear = nn.Linear(concat_size, cfg['HIDDEN_SIZE'])
        self.fc_logits = nn.Linear(cfg['HIDDEN_SIZE'], cfg['NUM_CLASSES'])
        # se_weight = torch.ones((3, cfg['HIDDEN_SIZE'])) * torch.tensor([1, 1, 1]).reshape(3, 1)
        # self.se_weight = nn.Parameter(se_weight)
        # self.title_dropout = nn.Dropout(p=cfg['TITLE_DROPOUT'])
        # self.video_dropout = nn.Dropout(p=cfg['VIDEO_DROPOUT'])
        self.fusion_dropout = nn.Dropout(p=cfg['VIDEO_DROPOUT'])
        self.se_fusion = nn.Sequential(
            nn.Linear(concat_size, concat_size // 2),
            nn.BatchNorm1d(concat_size // 2),
            nn.ReLU(inplace=True),
            nn.Linear(concat_size // 2, concat_size),
            #             nn.BatchNorm1d(d_text),
            nn.Sigmoid())

    def forward(self, video_feature, title_feature):
        """
        :param title_feature: title feature extracted from title representation
        :param label: classification target
        :return: (predictions, embeddings), model loss
        """
        concat_feature = torch.cat((video_feature, title_feature), dim=1)

        # 自注意力
        concat_feature = torch.mul(self.se_fusion(concat_feature), concat_feature)
        concat_feature = self.fusion_dropout(concat_feature)
        # 降维
        concat_hidden = self.concat_linear(concat_feature)

        normed_embedding = nn.functional.normalize(concat_hidden, p=2, dim=1)

        pred = self.fc_logits(torch.relu(concat_hidden))
        pred = torch.sigmoid(pred)
        return pred, normed_embedding


class Cls(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.fc_hidden = nn.Linear(cfg['INPUT_SIZE'], cfg['HIDDEN_SIZE'])
        self.fc_logits = nn.Linear(cfg['HIDDEN_SIZE'], cfg['NUM_CLASSES'])

    def forward(self, title_feature):
        embedding = self.fc_hidden(title_feature)
        normed_embedding = nn.functional.normalize(embedding, p=2, dim=1)
        pred = self.fc_logits(torch.relu(embedding))
        pred = torch.sigmoid(pred)
        return pred, normed_embedding


class BNCls(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        # nn.LayerNorm(cfg['INPUT_SIZE']),
        self.mlp_head = nn.Sequential(nn.Linear(cfg['INPUT_SIZE'], cfg['NUM_CLASSES']))

    def forward(self, embedding):
        normed_embedding = nn.functional.normalize(embedding, p=2, dim=1)
        pred = torch.sigmoid(self.mlp_head(embedding))
        return pred, normed_embedding


class PureCls(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.fc_logits = nn.Linear(cfg['INPUT_SIZE'], cfg['NUM_CLASSES'])

    def forward(self, title_feature):
        embedding = self.fc_logits(title_feature)
        normed_embedding = nn.functional.normalize(embedding, p=2, dim=1)
        pred = torch.sigmoid(embedding)
        return pred, normed_embedding


class AttentionCls(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.fc_align = nn.Linear(cfg['TITLE_INPUT_SIZE'], cfg['ALIGN_SIZE'])
        self.fc_attention1 = nn.Linear(cfg['ALIGN_SIZE'], cfg['HIDDEN_SIZE'] // cfg['REDUCE'])
        self.attention_bn = nn.BatchNorm1d(cfg['HIDDEN_SIZE'] // cfg['REDUCE'])
        self.fc_attention2 = nn.Linear(cfg['HIDDEN_SIZE'] // cfg['REDUCE'], cfg['HIDDEN_SIZE'])
        self.fc_logits = nn.Linear(cfg['HIDDEN_SIZE'], cfg['NUM_CLASSES'])

    def forward(self, video_feature, title_feature):
        title_feature = self.fc_align(title_feature)  # 768 to 1024
        attention_input = torch.add(title_feature, video_feature)
        attention = self.fc_attention1(attention_input)
        attention = self.attention_bn(attention)
        attention = self.fc_attention2(torch.relu(attention))
        attention = torch.sigmoid(attention)
        embedding = torch.add(torch.mul(title_feature, attention),
                              torch.mul(torch.sub(1, attention), video_feature))
        normed_embedding = nn.functional.normalize(embedding, p=2, dim=1)
        pred = self.fc_logits(torch.relu(embedding))
        pred = torch.sigmoid(pred)
        return pred, normed_embedding, embedding


class DimReduction(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.mlp_head = nn.Sequential(nn.Dropout(p=cfg['DROPOUT']), nn.LayerNorm(cfg['INPUT_SIZE']),
                                      nn.Linear(cfg['INPUT_SIZE'], cfg['OUTPUT_SIZE']))

    def forward(self, x):
        x = self.mlp_head(x)
        return x


class MyBertModel(TaskBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.bert = MacBert(cfg["MAC_BERT"])
        self.cls = Cls(cfg["BERT_CLASS"])
        self.model_list = [self.bert, self.cls]

    def forward(self, input_text):
        bert_output, bert_cls_hidden_state = self.bert(input_text)
        pred, normed_embedding = self.cls(bert_cls_hidden_state)
        return pred, normed_embedding


class MySingleBertModel(TaskBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.bert = MacBert(cfg["MAC_BERT"])
        self.cls = BNCls(cfg["SINGLE_BERT_CLASS"])
        self.model_list = [self.bert, self.cls]

    def forward(self, input_text):
        bert_output, bert_cls_hidden_state = self.bert(input_text)
        pred, normed_embedding = self.cls(bert_cls_hidden_state)
        return pred, normed_embedding


class MySingleNextVladModel(TaskBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.nextvlad = NeXtVLAD(cfg["NEXTVLAD"])
        self.cls = PureCls(cfg["SINGLE_NEXTVLAD_CLASS"])
        self.model_list = [self.nextvlad, self.cls]

    def forward(self, frame_feature):
        video_feature = self.nextvlad(frame_feature)
        pred, normed_embedding = self.cls(video_feature)
        return pred, normed_embedding


class MySingleTrsModel(TaskBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.trs = MyFrameFeatureTrs(cfg["FRAME_FEATURE_TRS"])
        self.cls = BNCls(cfg["FRAME_FEATURE_TRS_CLS"])
        self.model_list = [self.trs, self.cls]

    def forward(self, frame_feature, mask):
        frame_feature = frame_feature.cuda()
        mask = mask.cuda()
        _, video_embedding = self.trs(frame_feature, mask)
        pred, normed_embedding = self.cls(video_embedding)
        return pred, normed_embedding


class MyBertTrsSeModel(TaskBase):
    def __init__(self, cfg, bert):
        super().__init__(cfg)
        self.bert = bert
        self.trs = MyFrameFeatureTrs(cfg["FRAME_FEATURE_TRS"])
        self.fustion = MySeFustionAllModel(cfg['MySeFustionModel'])
        self.cls = BNCls(cfg["FUSION_CLS"])
        self.model_list = [self.bert, self.trs, self.fustion, self.cls]

    def forward(self, frame_feature, mask, title_asr):
        frame_feature = frame_feature.cuda()
        mask = mask.cuda()
        _, video_embedding = self.trs(frame_feature, mask)
        _, text_embedding = self.bert(title_asr)
        embedding = self.fustion(video_embedding, text_embedding)
        pred, normed_embedding = self.cls(embedding)
        return pred, normed_embedding


class MyNextvladModel(TaskBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.nextvlad = NeXtVLAD(cfg["NEXTVLAD"])
        self.cls = Cls(cfg["NeXtVLAD_CLASS"])
        self.model_list = [self.nextvlad, self.cls]

    def forward(self, input):
        video_feature = self.nextvlad(input)
        pred, normed_embedding = self.cls(video_feature)
        return pred, normed_embedding


class MyConcatModel(TaskBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.bert = MacBert(cfg["MAC_BERT"])
        self.nextvlad = NeXtVLAD(cfg["NEXTVLAD"])
        self.cls = Cls(cfg["CONCAT_CLASS"])
        self.model_list = [self.bert, self.nextvlad, self.cls]

    def forward(self, text_input, video_input):
        _, text_feature = self.bert(text_input)
        video_feature = self.nextvlad(video_input)
        feature = torch.cat([video_feature, text_feature], dim=1)
        pred, normed_embedding = self.cls(feature)
        return pred, normed_embedding


class MyConcatSeModel(TaskBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.bert = MacBert(cfg["MAC_BERT"])
        self.nextvlad = NeXtVLAD(cfg["NEXTVLAD"])
        self.cls = TitleVideoConcatSeCls(cfg["CONCAT_SE_CLASS"])
        self.model_list = [self.bert, self.nextvlad, self.cls]

    def forward(self, text_input, video_input):
        _, text_feature = self.bert(text_input)
        video_feature = self.nextvlad(video_input)
        pred, normed_embedding = self.cls(video_feature, text_feature)
        return pred, normed_embedding


class MyAttentionModel(TaskBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.bert = MacBert(cfg["MAC_BERT"])
        self.nextvlad = NeXtVLAD(cfg["NEXTVLAD"])
        self.cls = AttentionCls(cfg["ATTENTION_CLASS"])
        self.model_list = [self.bert, self.nextvlad, self.cls]

    def forward(self, text_input, video_input):
        _, text_feature = self.bert(text_input)
        video_feature = self.nextvlad(video_input)
        # feature = torch.cat([video_feature, text_feature], dim=1)
        pred, normed_embedding, embedding = self.cls(video_feature, text_feature)
        return pred, normed_embedding, embedding


class MyFinetuneModel(TaskBase):
    def __init__(self, cfg, pre_model_path):
        super().__init__(cfg)
        self.pretrain = torch.load(pre_model_path)
        self.model_list = self.pretrain.model_list

    def forward(self, text_input, video_input):
        _, normed_embedding = self.pretrain(text_input, video_input)
        b, n = normed_embedding.shape
        embedding_1, embedding_2 = normed_embedding[:b // 2], normed_embedding[b // 2:]
        cos = torch.mul(embedding_1, embedding_2)
        cos = torch.sum(cos, dim=1)
        return cos, normed_embedding


class MyTrsDirFinetuneModel(TaskBase):
    def __init__(self, cfg, pretrain_model):
        super().__init__(cfg)
        self.pretrain = pretrain_model
        self.dim_reduce = DimReduction(cfg['DIM_REDUCE'])
        self.model_list = [self.pretrain, self.dim_reduce]

    def forward(self, frame_feature, mask=None):
        frame_feature = frame_feature.cuda()
        if mask != None:
            mask = mask.cuda()
            embedding = self.pretrain(frame_feature, mask)
        else:
            embedding = self.pretrain(frame_feature)
        embedding = self.dim_reduce(embedding)
        normed_embedding = nn.functional.normalize(embedding, p=2, dim=1)
        b, n = normed_embedding.shape
        embedding_1, embedding_2 = normed_embedding[:b // 2], normed_embedding[b // 2:]
        cos = torch.mul(embedding_1, embedding_2)
        cos = torch.sum(cos, dim=1)
        return cos, normed_embedding
