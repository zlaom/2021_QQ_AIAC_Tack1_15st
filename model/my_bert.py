import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers as tfs
from model.model_base import ModelBase


class MacBert(ModelBase):
    def __init__(self, cfg):
        """
        :param cfg: concat_cls config defined in your_config.yaml
        """
        super().__init__(cfg)
        self.tokenizer = tfs.BertTokenizer.from_pretrained(cfg['MODEL_PATH'])
        self.bert = tfs.BertModel.from_pretrained(cfg['MODEL_PATH'])

    def forward(self, input_text):
        batch_tokenized = self.tokenizer(input_text,
                                         truncation=self.cfg['TRUNCATION'],
                                         padding=self.cfg['PADDING'],
                                         max_length=self.cfg['MAX_LENGTH'],
                                         return_tensors="pt")
        input_ids = batch_tokenized['input_ids'].cuda()
        token_type_ids = batch_tokenized['token_type_ids'].cuda()
        attention_mask = batch_tokenized['attention_mask'].cuda()
        bert_output = self.bert(input_ids=input_ids,
                                token_type_ids=token_type_ids,
                                attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:, 0, :]
        return bert_output[0], bert_cls_hidden_state