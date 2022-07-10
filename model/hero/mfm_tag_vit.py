from typing import Sequence
import torch
from torch import nn
import torch.nn.functional as F
import transformers
from model.hero.layers import (BertPooler, LinearLayer, MLPLayer,
                                BertLMPredictionHead,
                                BertEncoder)
from model.hero.embed import SubEmbeddings, ImageEmbeddings, FrameEmbeddings
from model.hero.encoder import RobertaPreTrainedModel, QueryFeatEncoder
from model.hero.model import VideoPreTrainedModel, FrameFeatureRegression, TagLinear
from model.hero.modeling_utils import mask_logits
from model.haoquan_transformers import VitEncoder

class MfmTagVitModel(nn.Module):
    def __init__(self, trs_config):
        super().__init__()
        # trs
        trs_cfg = trs_config["FRAME_FEATURE_TRS"]
        self.trs = VitEncoder(max_frame=trs_cfg['MAX_FRAME'],
                        frame_dim=trs_cfg['FRAME_DIM'],
                        dim=trs_cfg['DIM'],
                        depth=trs_cfg['DEPTH'],
                        heads=trs_cfg['HEADS'],
                        mlp_dim=trs_cfg['MLP_DIM'],
                        dim_head=trs_cfg['DIM_HEAD'],
                        dropout=trs_cfg['DROPOUT'],
                        emb_dropout=trs_cfg['EMB_DROPOUT'])
        # mfm
        self.frame_len = 32 # 不取cls token
        self.feat_regress = FrameFeatureRegression(1536, 1536) # 768->1536
        self.nce_temp = 1.0  # NCE shares same head with regression
        # self.mask_embedding = nn.Embedding(2, 1536, padding_idx=0)

        # tag used
        self.tag_loss = torch.nn.BCEWithLogitsLoss()
        self.cls = nn.Linear(1536, 10000)

    def forward(self, batch, task, compute_loss=True):
        if task == 'mfm':
            output = self.forward_mfm(batch, compute_loss)
        elif task == 'tag':
            output = self.forward_tag(batch, compute_loss)
        elif task == 'repr':
            output = self.forward_repr(batch)
        else:
            assert "wrong task name"
        return output

    def forward_mfm(self, batch, compute_loss=True):
        # 预处理，这里的预处理是在cpu上进行的
        mfm_mask = batch['mfm_mask']
        frame_feat = batch['frame_feature']

        mfm_label = self._compute_masked_hidden(frame_feat, mfm_mask) # 要在mask前保存真实标签
        frame_feat.masked_fill_(mfm_mask.unsqueeze(-1), 0) # 把mask的帧变为0

        # 正式开始，之后的都是涉及到gpu上的操作
        mfm_mask = mfm_mask.cuda()
        mfm_label = mfm_label.cuda()
        frame_feat = frame_feat.cuda()
        # mask = self.mask_embedding(mfm_mask.long()) # 加mask embedding
        # frame_feat = frame_feat + mask
        batch['frame_feature'] = frame_feat

        # 开始通过模型
        sequence_outputs = self.forward_repr(batch)
        frame_sequence_outputs = sequence_outputs[:, -self.frame_len:] # 取视频帧，后32帧

        # 取所有mask的帧的输出
        masked_output = self._compute_masked_hidden(frame_sequence_outputs, mfm_mask)
        prediction_feat = self.feat_regress(masked_output) # 投射到1536
        # 取负例，这里把所有的其他帧都看作负例了，包括pad
        neg_output = self._compute_masked_hidden(frame_sequence_outputs, ~mfm_mask)
        neg_pred_feat = self.feat_regress(neg_output)
        # nce loss
        if compute_loss:
            mfm_loss = self.mfm_nce(prediction_feat, mfm_label, neg_pred_feat)
            return mfm_loss
        else:
            return prediction_feat, mfm_label, neg_pred_feat


    def forward_tag(self, batch, compute_loss=True):
        frame_feat, frame_mask = self._compute_embeddings(batch)
        sequence_outputs = self.trs(frame_feat, frame_mask)
        pooled_output = sequence_outputs[:, 0]

        if compute_loss:
            tag_ids = batch['tag_id'].float().cuda()
            prediction_score = self.cls(pooled_output)
            loss = self.tag_loss(prediction_score, tag_ids)
            return loss
        else:
            normed_embedding = F.normalize(pooled_output, p=2, dim=1)
            return normed_embedding


    def forward_repr(self, batch):
        # embedding layer
        frame_feat, frame_mask = self._compute_embeddings(batch)
        sequence_outputs = self.trs(frame_feat, frame_mask)

        return sequence_outputs

    def _compute_embeddings(self, batch):
        frame_feat = batch['frame_feature'].cuda()
        frame_mask = batch['frame_mask'].cuda()
        return frame_feat, frame_mask

    def _compute_masked_hidden(self, hidden, mask):
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked

    def mfm_nce(self, masked_output, pos_output, neg_output, compute_loss=True):
        masked_score = masked_output.matmul(pos_output.t())
        # neg_score = masked_output.matmul(neg_output.t())
        # logits = torch.cat([masked_score, neg_score], dim=1).float()
        logits = masked_score.float()
        if compute_loss:
            targets = torch.arange(0, masked_output.size(0), dtype=torch.long, device=logits.device)
            loss = F.cross_entropy(logits/self.nce_temp, targets, reduction='mean')
            return loss
        else:
            return logits



