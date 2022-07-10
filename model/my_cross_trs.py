import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoConfig, AutoTokenizer, AutoModel, AutoModelForMaskedLM)
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from model.my_transformers import Transformer
from model.bmt.encoder import BiModalEncoder
from einops import rearrange, repeat


class FrameEmbedding2(nn.Module):
    """
    不降维
    """
    def __init__(self, config):
        super().__init__()
        # frame embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1536))
        # self.position_embeddings = nn.Conv2d(in_channels=1,
        #                                      out_channels=1,
        #                                      kernel_size=(5, 1),
        #                                      padding='same')
        # self.position_embeddings = nn.Embedding(64, 1536)
        self.LayerNorm = nn.LayerNorm(1536, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # 增加可学习class token
        B, _, _ = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x[:, :-1]  #保持原长度

        # 增加卷积position embeddings
        # x = x.unsqueeze(1)
        # position_embeddings = self.position_embeddings(x)
        # x = x + position_embeddings
        # x = x.squeeze()

        # 增加位置编码

        # position = torch.arange(32, device=x.device).repeat((B, 1))
        # position = self.position_embeddings(position)
        # x = x + position

        # x = self.LayerNorm(x)

        # x = self.dropout(x)

        return x


class MyCrossEncoder2(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        # text embedding
        self.word_embedding = AutoModel.from_pretrained(config['MODEL_PATHE'])
        # 解冻后3层
        unfreeze_layers = [f'layer.{i}' for i in [6, 7, 8, 9, 10, 11]]
        for name, param in self.word_embedding.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    break
        self.word_linear = nn.Linear(768, 1536)

        # frame embedding
        self.frame_embedding = FrameEmbedding2(config)
        self.frame_attention = Transformer(dim=1536,
                                           depth=6,
                                           heads=16,
                                           dim_head=96,
                                           mlp_dim=3072,
                                           dropout=0.1)

        # encoder
        self.encoder = Transformer(dim=1536,
                                   depth=6,
                                   heads=16,
                                   dim_head=96,
                                   mlp_dim=3072,
                                   dropout=0.1)

        # layer_morm
        self.layer_morm = nn.LayerNorm(1536)

    def forward(self,
                frame_features=None,
                frame_masks=None,
                input_ids=None,
                token_type_ids=None,
                text_masks=None):

        # 视频embedding Bx32x1536-> Bx32x1536
        # frame_embedding = frame_features
        frame_embedding = self.frame_embedding(frame_features)
        frame_embedding = self.frame_attention(frame_features, frame_masks)

        # 文本embedding Bx32x768
        text_embedding = self.word_embedding(input_ids,
                                             attention_mask=text_masks,
                                             token_type_ids=token_type_ids,
                                             output_hidden_states=True)
        text_embedding = text_embedding.hidden_states[-1]

        # 文本embedding Bx32x768 -> Bx32x1536
        text_embedding = self.word_linear(text_embedding)

        # 纵向cat embedding Bx32x1536+Bx32x1536->Bx64x1536
        cat_embedding = torch.cat((frame_embedding, text_embedding), dim=1)
        cat_masks = torch.cat((frame_masks, text_masks), dim=1)

        # layer_morm
        b, l, d = cat_embedding.shape
        cat_embedding = self.layer_morm(cat_embedding.reshape(-1, d)).view(b, l, d)

        # 通过cross encoder Bx64x1536 -> Bx64x1536
        cat_embedding = self.encoder(cat_embedding, cat_masks)

        return cat_embedding


# class MyCrossEncoder2(nn.Module):
#     def __init__(self, config) -> None:
#         super().__init__()

#         # 公共embedding
#         # self.type_embeddings = nn.Embedding(2, 1536, padding_idx=0)  # [frame:0, text:1]
#         self.layer_morm = nn.LayerNorm(1536)

#         # word embedding
#         # bert_config = AutoConfig.from_pretrained((config['MODEL_PATHE']))
#         # bert_config.hidden_size = 768
#         # bert_config.max_position_embeddings = 64
#         # bert_config.num_hidden_layers = 6
#         # self.word_embedding = AutoModel.from_config(bert_config)
#         # self.word_embedding = BertEmbeddings(bert_config)
#         self.word_embedding = AutoModel.from_pretrained(config['MODEL_PATHE'])
#         # self.word_attention = Transformer(dim=768,
#         #                                   depth=6,
#         #                                   heads=12,
#         #                                   dim_head=64,
#         #                                   mlp_dim=3072,
#         #                                   dropout=0.1)
#         self.word_linear = nn.Linear(768, 1536)
#         # 解冻后6层
#         unfreeze_layers = [f'layer.{i}' for i in [5, 6, 7, 8]]
#         for name, param in self.word_embedding.named_parameters():
#             param.requires_grad = False
#             for ele in unfreeze_layers:
#                 if ele in name:
#                     param.requires_grad = True
#                     break
#         # freeze_layers = [f'layer.{i}' for i in [6, 7, 8, 9, 10, 11]]
#         # for name, param in self.word_embedding.named_parameters():
#         #     # param.requires_grad = False
#         #     for ele in freeze_layers:
#         #         if ele in name:
#         #             param.requires_grad = False
#         #             break

#         # frame embedding
#         self.frame_embedding = FrameEmbedding2(config)
#         self.frame_attention = Transformer(dim=1536,
#                                            depth=4,
#                                            heads=12,
#                                            dim_head=128,
#                                            mlp_dim=3072,
#                                            dropout=0.1)

#         # encoder
#         self.encoder = Transformer(dim=1536,
#                                    depth=6,
#                                    heads=12,
#                                    dim_head=128,
#                                    mlp_dim=3072,
#                                    dropout=0.1)

#     def forward(self,
#                 frame_features=None,
#                 frame_masks=None,
#                 input_ids=None,
#                 token_type_ids=None,
#                 text_masks=None):

#         # 视频embedding Bx32x1536-> Bx32x1536
#         frame_embedding = self.frame_embedding(frame_features)
#         # frame_embedding = frame_features
#         frame_embedding = self.frame_attention(frame_embedding)

#         # 文本embedding Bx32x768
#         text_embedding = self.word_embedding(input_ids, output_hidden_states=True)
#         text_embedding = text_embedding.hidden_states[8]
#         # text_embedding = text_embedding.last_states
#         # text_embedding = self.word_embedding(input_ids)
#         # text_embedding = self.word_attention(text_embedding)

#         # 文本embedding Bx32x768 -> Bx32x1536
#         text_embedding = self.word_linear(text_embedding)

#         # 增加类别embedding
#         # frame_embedding, text_embedding = (
#         #     frame_embedding + self.type_embeddings(torch.full_like(frame_masks.long(), 0)),
#         #     text_embedding + self.type_embeddings(torch.full_like(text_masks.long(), 1)),
#         # )

#         # 纵向cat embedding Bx32x1536+Bx32x1536->Bx64x1536
#         cat_embedding = torch.cat((frame_embedding, text_embedding), dim=1)
#         cat_masks = torch.cat((frame_masks, text_masks), dim=1)

#         b, l, d = cat_embedding.shape
#         cat_embedding = self.layer_morm(cat_embedding.reshape(-1, d)).view(b, l, d)
#         # 通过cross encoder Bx64x1536 -> Bx64x1536
#         cat_embedding = self.encoder(cat_embedding, cat_masks)

#         # cat_embedding = self.layer_morm(cat_embedding)

#         return cat_embedding


class MyCrossFinetune2(nn.Module):
    def __init__(self, config, encoder):
        super().__init__()
        # self.encoder = MyCrossEncoder2(config)
        self.encoder = encoder
        # se
        self.f_fc = nn.Linear(1536, 256)
        self.t_fc = nn.Linear(1536, 256)
        self.f_se = nn.Sequential(nn.Linear(1536 * 2, 256), nn.BatchNorm1d(256), nn.Sigmoid())
        self.t_se = nn.Sequential(nn.Linear(1536 * 2, 256), nn.BatchNorm1d(256), nn.Sigmoid())

    def forward(self,
                frame_features=None,
                frame_masks=None,
                input_ids=None,
                token_type_ids=None,
                text_masks=None):
        cat_embedding = self.encoder(frame_features=frame_features,
                                     frame_masks=frame_masks,
                                     input_ids=input_ids,
                                     token_type_ids=token_type_ids,
                                     text_masks=text_masks)
        b, l, d = cat_embedding.shape
        f_embedding = cat_embedding[:, 0, :]
        t_embedding = cat_embedding[:, l // 2, :]
        f_t_embedding = torch.cat((f_embedding, t_embedding), dim=1)
        f_embedding = self.f_fc(f_embedding)
        t_embedding = self.t_fc(t_embedding)
        f_se = self.f_se(f_t_embedding)
        t_se = self.t_se(f_t_embedding)
        f_embedding = nn.functional.normalize(f_embedding, p=2, dim=1)
        t_embedding = nn.functional.normalize(t_embedding, p=2, dim=1)
        embedding = f_embedding * f_se + t_embedding * t_se

        normed_embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        b, n = normed_embedding.shape
        embedding_1, embedding_2 = normed_embedding[:b // 2], normed_embedding[b // 2:]
        cos = torch.mul(embedding_1, embedding_2)
        cos = torch.sum(cos, dim=1)
        return cos, normed_embedding


# class MyCrossFinetune2(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.encoder = MyCrossEncoder2(config)
#         self.fc = nn.Linear(1536 * 2, 256)

#     def forward(self,
#                 frame_features=None,
#                 frame_masks=None,
#                 input_ids=None,
#                 token_type_ids=None,
#                 text_masks=None):
#         cat_embedding = self.encoder(frame_features=frame_features,
#                                      frame_masks=frame_masks,
#                                      input_ids=input_ids,
#                                      token_type_ids=token_type_ids,
#                                      text_masks=text_masks)
#         b, l, d = cat_embedding.shape
#         f_embedding = cat_embedding[:, 0, :]
#         t_embedding = cat_embedding[:, l // 2, :]
#         f_t_embedding = torch.cat((f_embedding, t_embedding), dim=1)
#         embedding = self.fc(f_t_embedding)

#         normed_embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
#         b, n = normed_embedding.shape
#         embedding_1, embedding_2 = normed_embedding[:b // 2], normed_embedding[b // 2:]
#         cos = torch.mul(embedding_1, embedding_2)
#         cos = torch.sum(cos, dim=1)
#         return cos, normed_embedding

#     def init_weights(self):
#         """ Initialize the weights.
#         """
#         if isinstance(self, (nn.Linear, nn.Embedding)):
#             self.weight.data.normal_(mean=0.0, std=0.02)
#         elif isinstance(self, nn.LayerNorm):
#             self.bias.data.zero_()
#             self.weight.data.fill_(1.0)
#         if isinstance(self, nn.Linear) and self.bias is not None:
#             self.bias.data.zero_()


class MyCrossEncoder(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()

        # 公共embedding
        self.type_embeddings = nn.Embedding(2, 768, padding_idx=0)  # [frame:0, text:1]
        self.layer_morm = nn.LayerNorm(768)

        # word embedding
        bert_config = AutoConfig.from_pretrained((config['MODEL_PATHE']))
        self.word_embedding = BertEmbeddings(bert_config)

        # frame embedding
        self.frame_embedding = FrameEmbedding2(config)

        # encoder
        self.encoder = Transformer(dim=768,
                                   depth=12,
                                   heads=12,
                                   dim_head=64,
                                   mlp_dim=3072,
                                   dropout=0.1)

    def forward(self,
                frame_features=None,
                frame_masks=None,
                input_ids=None,
                token_type_ids=None,
                text_masks=None):

        # 对mask做处理 便于与cls token 结合
        if frame_masks is not None:
            frame_masks = F.pad(frame_masks.flatten(1), (1, 0), value=True)
            frame_masks = frame_masks[:, :-1]

        # 视频embedding Bx32x1536-> Bx32x768
        frame_embedding = self.frame_embedding(frame_features)

        # 文本embedding Bx32x768
        text_embedding = self.word_embedding(input_ids)

        # 增加类别embedding
        frame_embedding, text_embedding = (
            frame_embedding + self.type_embeddings(torch.full_like(frame_masks.long(), 0)),
            text_embedding + self.type_embeddings(torch.full_like(text_masks.long(), 1)),
        )

        # 纵向cat embedding Bx32x768+Bx32x768->Bx64x768
        cat_embedding = torch.cat((frame_embedding, text_embedding), dim=1)
        cat_masks = torch.cat((frame_masks, text_masks), dim=1)

        b, l, d = cat_embedding.shape
        cat_embedding = self.layer_morm(cat_embedding.reshape(-1, d)).view(b, l, d)
        # 通过cross encoder Bx64x768 -> Bx64x768
        cat_embedding = self.encoder(cat_embedding, cat_masks)

        # cat_embedding = self.layer_morm(cat_embedding)

        return cat_embedding


class MyCrossFinetune(nn.Module):
    def __init__(self, config, encoder, mlm_head_1):
        super().__init__()
        self.encoder = encoder
        self.fc = mlm_head_1

    def forward(self,
                frame_features=None,
                frame_masks=None,
                input_ids=None,
                token_type_ids=None,
                text_masks=None):
        cat_embedding = self.encoder(frame_features=frame_features,
                                     frame_masks=frame_masks,
                                     input_ids=input_ids,
                                     token_type_ids=token_type_ids,
                                     text_masks=text_masks)
        b, l, d = cat_embedding.shape
        # embedding = torch.cat((cat_embedding[:, 0, :], cat_embedding[:, l // 2, :]), dim=1)
        embedding = cat_embedding[:, l // 2, :]
        embedding = self.fc(embedding)
        normed_embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        b, n = normed_embedding.shape
        embedding_1, embedding_2 = normed_embedding[:b // 2], normed_embedding[b // 2:]
        cos = torch.mul(embedding_1, embedding_2)
        cos = torch.sum(cos, dim=1)
        return cos, normed_embedding

    def init_weights(self):
        """ Initialize the weights.
        """
        if isinstance(self, (nn.Linear, nn.Embedding)):
            self.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(self, nn.LayerNorm):
            self.bias.data.zero_()
            self.weight.data.fill_(1.0)
        if isinstance(self, nn.Linear) and self.bias is not None:
            self.bias.data.zero_()


class TreePretrain(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = MyCrossEncoder(config)
        # mlm 任务
        self.mlm_head_1 = nn.Linear(768, 256)
        self.mlm_head_2 = nn.Sequential(nn.LayerNorm(256, eps=1e-12, elementwise_affine=True),
                                        nn.Linear(256, 21128))
        self.mlm_loss = nn.CrossEntropyLoss()

        # tag 任务
        self.tag_emb = nn.Linear(768 * 2, 256)
        self.tag_head = nn.Sequential(nn.Linear(256, 10000))

        # tag se
        # self.tag_f = nn.Linear(768, 256)
        # self.tag_t = nn.Linear(768, 256)
        # self.tag_se_f = nn.Linear(768 * 2, 256)
        # self.tag_se_t = nn.Linear(768 * 2, 256)

        self.tag_loss = nn.BCEWithLogitsLoss()

        # FOM 任务
        self.fom_head_1 = nn.Linear(768, 256)
        self.fom_head_2 = nn.Sequential(nn.LayerNorm(256, eps=1e-12, elementwise_affine=True),
                                        nn.GELU(), nn.Linear(256, 32))
        self.fom_loss = nn.CrossEntropyLoss()

    def forward(self,
                jobs=None,
                frame_features=None,
                frame_masks=None,
                order_frame_features=None,
                order_labels=None,
                input_ids=None,
                token_type_ids=None,
                text_masks=None,
                word_masks=None,
                word_labels=None,
                tag_labels=None):

        # 对mask做处理 便于与cls token 结合
        if frame_masks is not None:
            frame_masks = F.pad(frame_masks.flatten(1), (1, 0), value=True)
            frame_masks = frame_masks[:, :-1]

        if order_labels is not None:
            order_labels = F.pad(order_labels.flatten(1), (1, 0), value=-1)  # 使用-1来表示cls token
            order_labels = order_labels[:, :-1]

        if 'fom' in jobs:
            frame_features = order_frame_features

        cat_embedding = self.encoder(frame_features=frame_features,
                                     frame_masks=frame_masks,
                                     input_ids=input_ids,
                                     token_type_ids=token_type_ids,
                                     text_masks=text_masks)

        _, S, _ = cat_embedding.shape

        frame_embedding, text_embedding = cat_embedding[:, :S // 2], cat_embedding[:, S // 2:]

        loss = 0
        losses = []
        tag_emb = None
        fom_emb = None

        sample = [0] * 4 + [1] * 1 + [2] * 2
        ch_job = random.choice(sample)

        record = {
            'mlm_loss': 0,
            'tag_loss': 0,
            'fom_loss': 0,
        }

        # mlm
        # if 'mlm' in jobs and word_labels is not None:
        if ch_job == 0 and word_labels is not None:
            mlm_emb = self.mlm_head_1(text_embedding)
            mlm_logits = self.mlm_head_2(mlm_emb)
            mlm_predict = mlm_logits[word_masks.ne(False)]
            mlm_labels = word_labels[word_masks.ne(False)]
            loss = self.mlm_loss(mlm_predict, mlm_labels)
            record['mlm_loss'] = loss.mean().item()
            emb = mlm_emb[:, 0]
            # losses.append(mlm_loss)
            # loss += mlm_loss

        # tag
        # if 'tag' in jobs and tag_labels is not None:
        if ch_job == 1 and tag_labels is not None:
            tag_emb = torch.cat((frame_embedding[:, 0], text_embedding[:, 0]), dim=-1)
            tag_emb = self.tag_emb(tag_emb)
            tag_logits = self.tag_head(tag_emb)
            loss = self.tag_loss(tag_logits, tag_labels.float())
            record['tag_loss'] = loss.mean().item()
            emb = tag_emb
            # losses.append(tag_loss)
            # loss += tag_loss

        # fom
        if 'fom' in jobs and order_labels is not None:
            # if ch_job == 2 and order_labels is not None:
            order_labels
            fom_emb = self.fom_head_1(frame_embedding)
            fom_logits = self.fom_head_2(fom_emb)
            fom_predict = fom_logits[order_labels.ne(-1)]
            fom_labels = order_labels[order_labels.ne(-1)]
            loss = self.fom_loss(fom_predict, fom_labels)
            record['fom_loss'] = loss.mean().item()
            emb = fom_emb[:, 0]
            # losses.append(fom_loss)
            # loss += fom_loss

        # record = {
        #     'mlm_loss': mlm_loss.mean().item(),
        #     'tag_loss': tag_loss.mean().item(),
        #     'fom_loss': fom_loss.mean().item(),
        # }
        # record = {
        #     'mlm_loss': mlm_loss.mean().item(),
        #     'tag_loss': 0,
        #     'fom_loss': fom_loss.mean().item(),
        # }

        normal_emb = nn.functional.normalize(emb, p=2, dim=1)
        # normal_emb = nn.functional.normalize(fom_emb[:, 0], p=2, dim=1)
        return (loss, normal_emb, record)

    def init_weights(self):
        """ Initialize the weights.
        """
        if isinstance(self, (nn.Linear, nn.Embedding)):
            self.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(self, nn.LayerNorm):
            self.bias.data.zero_()
            self.weight.data.fill_(1.0)
        if isinstance(self, nn.Linear) and self.bias is not None:
            self.bias.data.zero_()


class TagPretrain(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = MyCrossEncoder2(config)

        # tag se
        self.f_fc = nn.Linear(1536, 256)
        self.t_fc = nn.Linear(1536, 256)
        self.f_se = nn.Linear(1536 * 2, 256)
        self.t_se = nn.Linear(1536 * 2, 256)
        self.tag_head = nn.Sequential(nn.Linear(256, 10000))
        self.tag_loss = nn.BCEWithLogitsLoss()

    def forward(self,
                frame_features=None,
                frame_masks=None,
                input_ids=None,
                token_type_ids=None,
                text_masks=None,
                tag_labels=None):

        cat_embedding = self.encoder(frame_features=frame_features,
                                     frame_masks=frame_masks,
                                     input_ids=input_ids,
                                     token_type_ids=token_type_ids,
                                     text_masks=text_masks)

        loss = 0

        record = {
            'mlm_loss': 0,
            'tag_loss': 0,
            'fom_loss': 0,
        }

        b, l, d = cat_embedding.shape
        f_embedding = cat_embedding[:, 0, :]
        t_embedding = cat_embedding[:, l // 2, :]
        f_t_embedding = torch.cat((f_embedding, t_embedding), dim=1)
        f_embedding = self.f_fc(f_embedding)
        t_embedding = self.t_fc(t_embedding)
        f_se = self.f_se(f_t_embedding)
        t_se = self.t_se(f_t_embedding)
        f_embedding = nn.functional.normalize(f_embedding, p=2, dim=1)
        t_embedding = nn.functional.normalize(t_embedding, p=2, dim=1)
        embedding = f_embedding * f_se + t_embedding * t_se

        tag_logits = self.tag_head(embedding)

        loss = self.tag_loss(tag_logits, tag_labels.float())
        record['tag_loss'] = loss.mean().item()
        normed_embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        return (loss, normed_embedding, record)

    def init_weights(self):
        """ Initialize the weights.
        """
        if isinstance(self, (nn.Linear, nn.Embedding)):
            self.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(self, nn.LayerNorm):
            self.bias.data.zero_()
            self.weight.data.fill_(1.0)
        if isinstance(self, nn.Linear) and self.bias is not None:
            self.bias.data.zero_()