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

class MlmTagBertModel(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        # self.encoder = BertEncoder(config)
        # self.embeddings = SubEmbeddings(config)

        # self.pooler = BertPooler(config)
        # self.apply(self.init_weights)
        self.config = config

        self.bert = transformers.BertModel.from_pretrained('pretrain/macbert')
        # pretraining
        # self.lm_head = BertLMPredictionHead(config, self.embeddings.word_embeddings.weight)
        self.lm_head = BertLMPredictionHead(config, self.bert.embeddings.word_embeddings.weight)

        # tag used
        self.tag_loss = torch.nn.BCEWithLogitsLoss()
        # self.cls = TagLinear(config.hidden_size)
        self.cls = nn.Linear(768, 10000)

    def forward(self, batch, task, compute_loss=True):
        if task == 'mlm':
            output = self.forward_mlm(batch, compute_loss)
        elif task == 'tag':
            output = self.forward_tag(batch, compute_loss)
        elif task == 'repr':
            output = self.forward_repr(batch)
        else:
            assert "wrong task name"
        return output


    def forward_mlm(self, batch, compute_loss=True):
        input_ids, attention_mask, token_type_ids = self._compute_embeddings(batch)
        sequence_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]

        mlm_labels = batch['mlm_labels']
        mlm_title_mask = (mlm_labels != -1)
        mlm_labels = mlm_labels[mlm_title_mask].contiguous().view(-1)

        mlm_labels = mlm_labels.cuda()
        mlm_title_mask = mlm_title_mask.cuda()

        masked_output = self._compute_masked_hidden(sequence_output, mlm_title_mask)
        prediction_scores = self.lm_head(masked_output)

        if compute_loss:
            masked_lm_loss = F.cross_entropy(prediction_scores, mlm_labels, reduction='mean')
            return masked_lm_loss
        else:
            return prediction_scores.cpu(), mlm_labels.cpu()


    def forward_tag(self, batch, compute_loss=True):
        input_ids, attention_mask, token_type_ids = self._compute_embeddings(batch)
        sequence_outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[0]

        # pooled_output = self.pooler(sequence_outputs)
        pooled_output = sequence_outputs[:, 0] # cls token

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
        input_ids, attention_mask, token_type_ids = self._compute_embeddings(batch)
        encoder_outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        sequence_output = encoder_outputs[0]
        # pooled_output = self.pooler(sequence_output)
        # output = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return sequence_output


    def _compute_embeddings(self, batch):
        input_ids=batch['input_ids'].cuda()
        title_mask = batch['title_mask'].cuda()
        token_type_ids = batch['token_type_ids'].cuda()
        # token_type_id
        # device = frame_mask.device
        # frame_type_embeddings = self.embeddings.token_type_embeddings(torch.ones(1, 1, dtype=torch.long, device=device))
        # frame_embedding = self.img_embeddings(frame_feat, frame_type_embeddings) # 单纯的投射关系 + pos_embed [+ type_embed]

        return input_ids, title_mask, token_type_ids

    def _compute_masked_hidden(self, hidden, mask):
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked
        