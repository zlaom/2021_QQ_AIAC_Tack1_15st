import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers as tfs
from model.bmt.encoder import BiModalEncoder
from model.my_transformers import FrameFeatureTrs
from model.model_base import TaskBase, ModelBase


class MyBmt(ModelBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.bmt = BiModalEncoder(d_model_A=768,
                                  d_model_V=1536,
                                  d_model=None,
                                  dout_p=0.1,
                                  H=4,
                                  d_ff_A=4 * 768,
                                  d_ff_V=4 * 1536,
                                  N=2)

    def forward(self, text_hidden, text_mask, frame_feature, frame_mask):
        T, F = self.bmt((text_hidden, frame_feature), (text_mask, frame_mask))
        return (T, F)


class MyBmtDemo1(TaskBase):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.tokenizer = tfs.BertTokenizer.from_pretrained('data/chinese_macbert_base')
        self.bert = tfs.BertModel.from_pretrained('data/chinese_macbert_base')
        self.bmt = BiModalEncoder(d_model_A=768,
                                  d_model_V=1536,
                                  d_model=None,
                                  dout_p=0.1,
                                  H=4,
                                  d_ff_A=4 * 768,
                                  d_ff_V=4 * 1536,
                                  N=2)
        self.fr_trs = FrameFeatureTrs()
        self.emd_head = nn.Sequential(nn.Linear(768 + 1536, 256), nn.ReLU())
        self.cls_head = nn.Sequential(nn.Linear(256, 10000), nn.Sigmoid())

    def forward(self, frame_feature, frame_mask, title_asr):
        frame_feature = frame_feature.cuda()
        frame_mask = frame_mask.cuda()

        # 文本self attention
        batch_tokenized = self.tokenizer(title_asr,
                                         truncation=True,
                                         padding='max_length',
                                         max_length=64,
                                         return_tensors="pt")
        input_ids = batch_tokenized['input_ids'].cuda()
        token_type_ids = batch_tokenized['token_type_ids'].cuda()
        attention_mask = batch_tokenized['attention_mask'].cuda()
        bert_output = self.bert(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)

        # 视频self attention
        fr_trs_output, _ = self.fr_trs(frame_feature, frame_mask)

        # Bmt
        b, d = attention_mask.shape
        attention_mask = attention_mask.reshape((b, 1, d))
        frame_mask = F.pad(frame_mask.flatten(1), (1, 0), value=True)
        b, d = frame_mask.shape
        frame_mask = frame_mask.reshape((b, 1, d))
        t_a_f, f_a_t = self.bmt((bert_output[0], fr_trs_output), (attention_mask, frame_mask))

        # embedding
        t_a_f_emb = t_a_f[:, 0, :]
        f_a_t_emb = f_a_t[:, 0, :]
        embedding = self.emd_head(torch.cat((t_a_f_emb, f_a_t_emb), dim=-1))

        # predict
        pre = self.cls_head(embedding)

        normed_embedding = nn.functional.normalize(embedding, p=2, dim=1)

        return pre, normed_embedding
