from math import isnan
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoConfig, AutoTokenizer, AutoModel, AutoModelForMaskedLM)
from model.my_transformers import Transformer
from model.bmt.encoder import BiModalEncoder
from einops import rearrange, repeat


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""
    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz, eps=1e-5)
        layers = [nn.Dropout(dropout), nn.Linear(in_hsz, out_hsz)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


class TextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.padding_idx = 1
        self.word_embeddings = nn.Embedding(config['VOCAB_SIZE'],
                                            config['HIDDEN_SIZE'],
                                            padding_idx=self.padding_idx)
        self.position_embeddings = nn.Embedding(config['MAX_POSITION_EMBEDDINGS'],
                                                config['HIDDEN_SIZE'])
        self.token_type_embeddings = nn.Embedding(config['VOCAB_SIZE'], config['HIDDEN_SIZE'])
        self.LayerNorm = nn.LayerNorm(config['HIDDEN_SIZE'], eps=1e-5)
        self.dropout = nn.Dropout(config['HIDDEN_DROPOUT_PROB'])

    def forward(self, input_ids=None, position_ids=None, token_type_ids=None, inputs_embeds=None):
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if position_ids is None:
            if input_ids is not None:
                # Create the position ids from the input token ids.
                # Any padded tokens remain padded.
                position_ids = self.create_position_ids_from_input_ids(input_ids).to(device)
            else:
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        position_embeddings = self.position_embeddings(position_ids)
        if token_type_ids is None:
            token_type_embeddings = self.token_type_embeddings(
                torch.ones(1, 1, dtype=torch.long, device=device))
        else:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = (inputs_embeds + position_embeddings + token_type_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_input_ids(self, x):
        """ Replace non-padding symbols with their position numbers.
            Position numbers begin at padding_idx+1.
            Padding symbols are ignored.
            This is modified from fairseq's `utils.make_positions`.
        :param torch.Tensor x:
        :return torch.Tensor:
        """
        mask = x.ne(self.padding_idx).long()
        incremental_indicies = torch.cumsum(mask, dim=1) * mask
        return incremental_indicies + self.padding_idx

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """ We are provided embeddings directly.
            We cannot infer which are padded so just generate
            sequential position ids.
        :param torch.Tensor inputs_embeds:
        :return torch.Tensor:
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(self.padding_idx + 1,
                                    sequence_length + self.padding_idx + 1,
                                    dtype=torch.long,
                                    device=inputs_embeds.device)
        return position_ids.unsqueeze(0).expand(input_shape)


class FrameEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.frame_linear = nn.Linear(config['FRAME_DIM'], config['HIDDEN_SIZE'])
        self.frame_layer_norm = nn.LayerNorm(config['FRAME_DIM'], eps=1e-5)
        self.position_embeddings = nn.Embedding(config['MAX_FRAME_SEQ_LEN'], config['HIDDEN_SIZE'])
        self.mask_embedding = nn.Embedding(2, config['FRAME_DIM'], padding_idx=0)

        # tf naming convention for layer norm
        self.LayerNorm = nn.LayerNorm(config['HIDDEN_SIZE'])
        self.dropout = nn.Dropout(config['HIDDEN_DROPOUT_PROB'])

    def forward(self, frame_feat, type_embeddings=None, frame_pos_ids=None, frame_masks=None):
        if frame_pos_ids is None:
            frame_pos_ids = self.create_position_ids_from_inputs_embeds(frame_feat)

        if frame_masks is not None:
            mask = self.mask_embedding(frame_masks.long())
            frame_feat = frame_feat + mask

        transformed_im = self.frame_linear(self.frame_layer_norm(frame_feat))
        position_embeddings = self.position_embeddings(frame_pos_ids)
        embeddings = transformed_im + position_embeddings
        if type_embeddings is not None:
            embeddings += type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """ We are provided embeddings directly.
            We cannot infer which are padded so just generate
            sequential position ids.
        :param torch.Tensor inputs_embeds:
        :return torch.Tensor:
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        position_ids = torch.arange(0,
                                    sequence_length,
                                    dtype=torch.long,
                                    device=inputs_embeds.device)
        return position_ids.unsqueeze(0).expand(input_shape)


class MyCrossMLM(nn.Module):
    def __init__(self, text_embedding, frame_embedding, encoder, cls) -> None:
        super(MyCrossMLM, self).__init__()
        self.text_embedding = text_embedding
        self.frame_embedding = frame_embedding
        self.encoder = encoder
        self.cls = cls

    def forward(self,
                frame_feature=None,
                frame_masks=None,
                input_ids=None,
                token_type_ids=None,
                text_mask=None):

        # 视频embedding Bx32x1536-> Bx32x768
        f_embedding = self.frame_embedding(frame_feat=frame_feature, frame_masks=frame_masks)

        # 文本embedding Bx32x768
        t_embedding = self.text_embedding(input_ids=input_ids, token_type_ids=token_type_ids)

        # 纵向cat embedding Bx32x768+Bx32x768->Bx64x768
        cat_embedding = torch.cat((f_embedding, t_embedding), dim=1)
        cat_masks = torch.cat((frame_masks, text_mask), dim=1)

        # 通过cross encoder Bx64x768 -> Bx64x768
        f_embedding = self.encoder(inputs_embeds=cat_embedding,
                                   attention_mask=cat_masks).last_hidden_state

        # 取出text embedding Bx32x768
        f_t_embedding = f_embedding[:, 32:, :]

        # Bx32x768 -> Bx32x2w
        logits = self.cls(f_t_embedding)

        return (logits, f_t_embedding)

        # # 取出frame embedding Bx32x768
        # f_f_embedding = f_embedding[:, :32, :]

        # # Bx32x768 -> Bx32x2w
        # logits = self.cls(f_f_embedding)

        # return (logits, f_f_embedding)


class MyCrossMLM2(nn.Module):
    def __init__(self, config) -> None:
        super(MyCrossMLM2, self).__init__()
        # 公共embedding
        self.type_embeddings = nn.Embedding(3, 768, padding_idx=0)  # [frame:2, title:0, asr:1]
        self.layer_morm = nn.LayerNorm(768, eps=1e-12, elementwise_affine=True)

        # word embedding
        self.word_embedding = nn.Embedding(21128, 768, padding_idx=0)
        self.word_position_embedding = nn.Embedding(256, 768, padding_idx=0)

        # frame embedding
        self.frame_conv = nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=(6, 1),
                                    padding='same')
        self.frame_embedding = nn.Sequential(nn.LayerNorm(1536, eps=1e-5), nn.Linear(1536, 768))

        # encoder
        self.encoder = Transformer(dim=768,
                                   depth=6,
                                   heads=12,
                                   dim_head=64,
                                   mlp_dim=3072,
                                   dropout=0.1)

        # transform
        self.transform = nn.Sequential(nn.Linear(768, 256),
                                       nn.LayerNorm(256, eps=1e-12, elementwise_affine=True))

        # decoder
        self.decoder = nn.Linear(256, 21128)

    def forward(self,
                frame_feature=None,
                frame_masks=None,
                input_ids=None,
                token_type_ids=None,
                text_mask=None):

        # 视频embedding Bx32x1536-> Bx32x768
        # 视频降维
        f_embedding = self.frame_embedding(frame_feature)

        # 视频卷积位置编码
        f_embedding = f_embedding.unsqueeze(1)
        f_embedding = f_embedding + self.frame_conv(f_embedding)
        f_embedding = f_embedding.squeeze()

        # 文本embedding Bx32x768
        t_embedding = self.word_embedding(input_ids)
        # 文本位置编码
        t_position = torch.arange(t_embedding.shape[1], dtype=torch.long,
                                  device=t_embedding.device).unsqueeze(0).repeat(
                                      (t_embedding.shape[0], 1))
        t_embedding = t_embedding + self.word_position_embedding(t_position)

        # 纵向cat embedding Bx32x768+Bx32x768->Bx64x768
        cat_embedding = torch.cat((f_embedding, t_embedding), dim=1)
        cat_masks = torch.cat((frame_masks, text_mask), dim=1)

        # type_embeddings video title asr
        b, s, d = f_embedding.shape
        cat_type_ids = torch.cat((torch.ones(
            (b, s), device=t_embedding.device) * 2, token_type_ids),
                                 dim=-1).long()
        cat_embedding = cat_embedding + self.type_embeddings(cat_type_ids)
        cat_embedding = self.layer_morm(cat_embedding)

        # 通过cross encoder Bx64x768 -> Bx64x768
        f_embedding = self.encoder(cat_embedding, cat_masks)

        # 取出text embedding Bx32x768
        f_t_embedding = f_embedding[:, 32:, :]

        # get embedding Bx32x768 -> Bx32x256
        f_t_embedding = self.transform(f_t_embedding)

        # get logits Bx32x256 -> Bx32x2w
        logits = self.decoder(f_t_embedding)

        return (logits, f_t_embedding)

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


"""
实验一:  去掉type token 0.639
实验二： 不压缩视频 直接传bmt 0.6126
实验三： 单视频trs
"""


class MySingleTrs(nn.Module):
    def __init__(self, config) -> None:
        super(MySingleTrs, self).__init__()
        # frame embedding
        self.frame_class_token = nn.Parameter(torch.randn(1, 1, 1536))
        self.frame_conv = nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=(6, 1),
                                    padding='same')

        # encoder
        self.encoder = Transformer(dim=1536,
                                   depth=6,
                                   heads=12,
                                   dim_head=1536 // 12,
                                   mlp_dim=1536 * 4,
                                   dropout=0.1)

        # transform
        self.transform = nn.Sequential(nn.Linear(1536, 256))

    def forward(self,
                frame_feature=None,
                frame_masks=None,
                input_ids=None,
                token_type_ids=None,
                text_mask=None):

        # 视频embedding Bx32x1536
        f_embedding = frame_feature

        # 添加class token
        b, n, _ = f_embedding.shape
        frame_class_token = repeat(self.frame_class_token, '() n d -> b n d', b=b)
        f_embedding = torch.cat((frame_class_token, f_embedding), dim=1)
        f_embedding = f_embedding[:, :32, :]
        frame_masks = F.pad(frame_masks.flatten(1), (1, 0), value=True)
        frame_masks = frame_masks[:, :32]

        # 视频卷积位置编码
        f_embedding = f_embedding.unsqueeze(1)
        f_embedding = f_embedding + self.frame_conv(f_embedding)
        f_embedding = f_embedding.squeeze()

        # 通过cross encoder Bx64x768 -> Bx64x768
        f_embedding = self.encoder(f_embedding, frame_masks)

        # 取出text embedding Bx32x768
        f_t_embedding_max = torch.max(f_embedding, dim=1)[0]
        f_t_embedding_mean = torch.mean(f_embedding, dim=1)
        # f_t_embedding = f_embedding[:, 0, :]
        f_t_embedding = f_t_embedding_max * 0.2 + f_t_embedding_mean

        # get embedding Bx32x768 -> Bx32x256
        f_t_embedding = self.transform(f_t_embedding)

        return (None, f_t_embedding)

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


class MyBmtMLM(nn.Module):
    def __init__(self, config) -> None:
        super(MyBmtMLM, self).__init__()
        # 公共embedding
        self.type_embeddings = nn.Embedding(3, 768, padding_idx=0)  # [frame:2, title:0, asr:1]
        self.layer_morm = nn.LayerNorm(768, eps=1e-12, elementwise_affine=True)

        # word embedding
        self.word_embedding = nn.Embedding(21128, 768, padding_idx=0)
        self.word_position_embedding = nn.Embedding(256, 768, padding_idx=0)

        # frame embedding
        self.frame_class_token = nn.Parameter(torch.randn(1, 1, 1536))

        self.frame_conv = nn.Conv2d(in_channels=1,
                                    out_channels=1,
                                    kernel_size=(6, 1),
                                    padding='same')

        self.frame_embedding = nn.Sequential(nn.LayerNorm(1536, eps=1e-5), nn.Linear(1536, 768))

        # encoder
        self.encoder = BiModalEncoder(d_model_A=1536,
                                      d_model_V=768,
                                      d_model=None,
                                      H=12,
                                      d_ff_A=4608,
                                      d_ff_V=3072,
                                      N=3,
                                      dout_p=0.1)

        # transform
        self.transform = nn.Sequential(nn.Dropout(0.1), nn.BatchNorm1d(1536 + 768),
                                       nn.Linear(1536 + 768, 256))

        # decoder
        self.decoder = nn.Linear(256, 21128)

    def forward(self,
                frame_feature=None,
                frame_masks=None,
                input_ids=None,
                token_type_ids=None,
                text_mask=None):

        # 视频embedding Bx32x1536-> Bx32x768
        # 视频降维
        # f_embedding = self.frame_embedding(frame_feature)

        f_embedding = frame_feature

        # 添加class token
        b, n, _ = f_embedding.shape
        frame_class_token = repeat(self.frame_class_token, '() n d -> b n d', b=b)
        f_embedding = torch.cat((frame_class_token, f_embedding), dim=1)
        f_embedding = f_embedding[:, :32, :]
        frame_masks = F.pad(frame_masks.flatten(1), (1, 0), value=True)
        frame_masks = frame_masks[:, :32]

        # 视频卷积位置编码
        f_embedding = f_embedding.unsqueeze(1)
        f_embedding = f_embedding + self.frame_conv(f_embedding)
        f_embedding = f_embedding.squeeze()

        # 文本embedding Bx32x768
        t_embedding = self.word_embedding(input_ids)
        # 文本位置编码
        t_position = torch.arange(t_embedding.shape[1], dtype=torch.long,
                                  device=t_embedding.device).unsqueeze(0).repeat(
                                      (t_embedding.shape[0], 1))
        t_embedding = t_embedding + self.word_position_embedding(t_position)

        # 通过BMT
        f_a_t_embedding, t_a_f_embedding = self.encoder((f_embedding, t_embedding),
                                                        (frame_masks, text_mask))
        # 纵向拼接 Bx1x1536
        f_t_embedding = torch.cat((f_a_t_embedding[:, 0], t_a_f_embedding[:, 0]), dim=-1)

        # 降维 Bx32x1536 -> Bx32x256
        f_t_embedding = self.transform(f_t_embedding)

        # get logits Bx32x256 -> Bx32x2w
        logits = self.decoder(f_t_embedding)

        return (logits, nn.functional.normalize(f_t_embedding, p=2, dim=1))

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


# ============ 版本1 ==================
# class MyBmtMLM(nn.Module):
#     def __init__(self, config) -> None:
#         super(MyBmtMLM, self).__init__()
#         # 公共embedding
#         self.type_embeddings = nn.Embedding(3, 768, padding_idx=0)  # [frame:2, title:0, asr:1]
#         self.layer_morm = nn.LayerNorm(768, eps=1e-12, elementwise_affine=True)

#         # word embedding
#         self.word_embedding = nn.Embedding(21128, 768, padding_idx=0)
#         self.word_position_embedding = nn.Embedding(256, 768, padding_idx=0)

#         # frame embedding
#         self.frame_class_token = nn.Parameter(torch.randn(1, 1, 768))

#         self.frame_conv = nn.Conv2d(in_channels=1,
#                                     out_channels=1,
#                                     kernel_size=(6, 1),
#                                     padding='same')

#         self.frame_embedding = nn.Sequential(nn.LayerNorm(1536, eps=1e-5), nn.Linear(1536, 768))

#         # encoder
#         self.encoder = BiModalEncoder(d_model_A=768,
#                                       d_model_V=768,
#                                       d_model=None,
#                                       H=12,
#                                       d_ff_A=3072,
#                                       d_ff_V=3072,
#                                       N=3,
#                                       dout_p=0.1)

#         # transform
#         self.transform = nn.Sequential(nn.Dropout(0.1), nn.BatchNorm1d(1536), nn.Linear(1536, 256))

#         # decoder
#         self.decoder = nn.Linear(256, 21128)

#     def forward(self,
#                 frame_feature=None,
#                 frame_masks=None,
#                 input_ids=None,
#                 token_type_ids=None,
#                 text_mask=None):

#         # 视频embedding Bx32x1536-> Bx32x768
#         # 视频降维
#         f_embedding = self.frame_embedding(frame_feature)

#         # 添加class token
#         b, n, _ = f_embedding.shape
#         frame_class_token = repeat(self.frame_class_token, '() n d -> b n d', b=b)
#         f_embedding = torch.cat((frame_class_token, f_embedding), dim=1)
#         f_embedding = f_embedding[:, :32, :]
#         frame_masks = F.pad(frame_masks.flatten(1), (1, 0), value=True)
#         frame_masks = frame_masks[:, :32]

#         # 视频卷积位置编码
#         f_embedding = f_embedding.unsqueeze(1)
#         f_embedding = f_embedding + self.frame_conv(f_embedding)
#         f_embedding = f_embedding.squeeze()

#         # 文本embedding Bx32x768
#         t_embedding = self.word_embedding(input_ids)
#         # 文本位置编码
#         t_position = torch.arange(t_embedding.shape[1], dtype=torch.long,
#                                   device=t_embedding.device).unsqueeze(0).repeat(
#                                       (t_embedding.shape[0], 1))
#         t_embedding = t_embedding + self.word_position_embedding(t_position)

#         # type_embeddings video title asr
#         # 纵向cat embedding Bx32x768+Bx32x768->Bx64x768
#         cat_embedding = torch.cat((f_embedding, t_embedding), dim=1)

#         b, s, d = f_embedding.shape
#         cat_type_ids = torch.cat((torch.ones(
#             (b, s), device=t_embedding.device) * 2, token_type_ids),
#                                  dim=-1).long()
#         cat_embedding = cat_embedding + self.type_embeddings(cat_type_ids)
#         cat_embedding = self.layer_morm(cat_embedding)

#         # 通过BMT
#         f_embedding, t_embedding = cat_embedding[:, :32], cat_embedding[:, 32:]
#         f_a_t_embedding, t_a_f_embedding = self.encoder((f_embedding, t_embedding),
#                                                         (frame_masks, text_mask))
#         # 纵向拼接 Bx1x1536
#         f_t_embedding = torch.cat((f_a_t_embedding[:, 0], t_a_f_embedding[:, 0]), dim=-1)

#         # 降维 Bx32x1536 -> Bx32x256
#         f_t_embedding = self.transform(f_t_embedding)

#         # get logits Bx32x256 -> Bx32x2w
#         logits = self.decoder(f_t_embedding)

#         return (logits, nn.functional.normalize(f_t_embedding, p=2, dim=1))

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

## ViLT


class Se(nn.Module):
    def __init__(self, feature_1, feature_2):
        super().__init__()
        self.se = nn.Sequential(nn.Linear(feature_1, feature_2), nn.BatchNorm1d(feature_2),
                                nn.Sigmoid())

    def forward(self, features):
        return self.se(features)


class BertMLM(nn.Module):
    def __init__(self, config) -> None:
        super(BertMLM, self).__init__()

        feature_size = 1536
        hidden_size = 256
        video_size = 1536
        title_size = 768

        self.title_encoder = AutoConfig.from_pretrained((config['MODEL_PATHE']))
        self.cross_encoder = AutoConfig.from_pretrained((config['MODEL_PATHE']))
        self.linear = nn.Linear(title_size, video_size)
        self.ln = nn.LayerNorm(video_size)
        self.fc2 = nn.Linear(video_size, hidden_size)
        self.fc3 = nn.Linear(video_size, hidden_size)
        self.se1 = Se(video_size * 2, hidden_size)
        self.se2 = Se(video_size * 2, hidden_size)

    def forward(self, video_feature, frame_masks, title_feature):
        title_feature, title_mask = self.title_encoder(title_feature)
        title_feature = self.linear(title_feature)
        cat_features = torch.cat((video_feature, title_feature), dim=1)
        cat_masks = torch.cat((frame_masks, title_mask), dim=1)
        b, l, dim = cat_features.shape

        cat_features = self.ln(cat_features.reshape(-1, dim)).view(b, l, dim)

        cross_features = self.cross_encoder(cat_features, cat_masks)
        b, l, dim = cross_features.shape

        frame_first = torch.cat((cross_features[:, 0, :], cross_features[:, l // 2, :]), dim=1)
        video_first, title_first = cross_features[:, 0, :], cross_features[:, l // 2, :]
        video_weight, title_weight = self.se1(frame_first), self.se2(frame_first)
        video_first, title_first = self.fc2(video_first), self.fc3(title_first)
        video_first = torch.nn.functional.normalize(video_first, p=2, dim=1)
        title_first = torch.nn.functional.normalize(title_first, p=2, dim=1)
        embedding = video_first * video_weight + title_first * title_weight
        normed_embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        b, n = normed_embedding.shape
        embedding_1, embedding_2 = normed_embedding[:b // 2], normed_embedding[b // 2:]
        cos = torch.mul(embedding_1, embedding_2)
        cos = torch.sum(cos, dim=1)
        return cos, normed_embedding