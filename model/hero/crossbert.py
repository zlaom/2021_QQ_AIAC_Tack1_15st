import torch
from torch import nn
import torch.nn.functional as F
import transformers
from model.hero.layers import (BertPooler, LinearLayer, MLPLayer,
                                BertLMPredictionHead,
                                BertEncoder)
from model.hero.embed import SubEmbeddings, ImageEmbeddings, FrameEmbeddings
from model.hero.encoder import RobertaPreTrainedModel
from model.hero.model import VideoPreTrainedModel, FrameFeatureRegression

class CrossModalTrm(RobertaPreTrainedModel):

    def __init__(self, config, sep_config, mix_config, frame_dim=1536, max_frame_seq_len=32):
        super().__init__(config)
        self.txt_encoder = BertEncoder(sep_config)
        self.frame_encoder = BertEncoder(sep_config)
        self.mix_encoder = BertEncoder(mix_config)

        self.embeddings = SubEmbeddings(config)
        
        self.img_embeddings = ImageEmbeddings(config, frame_dim, max_frame_seq_len)
        self.pooler = BertPooler(config)
        self.apply(self.init_weights)
        self.config = config

        # pretraining
        self.lm_head = BertLMPredictionHead(
            config, self.embeddings.word_embeddings.weight)

    def forward(self, batch, task='mlm_train'):
        if task.startswith('mlm'):
            output = self.forward_mlm(batch, task)
        elif task == 'repr':
            output = self.forward_repr(batch)
        else:
            assert "wrong task name"
        return output

    def forward_mlm(self, batch, task='mlm_train'):
        title_embedding, title_mask, frame_embedding, frame_mask = self._compute_sep_embeddings(batch)
        title_sequence_output = self.txt_encoder(title_embedding, title_mask)[0]
        frame_sequence_output = self.frame_encoder(frame_embedding, frame_mask)[0]
        title_sequence_output = title_sequence_output + title_embedding
        frame_sequence_output = frame_sequence_output + frame_embedding
        embedding = torch.cat([title_sequence_output, frame_sequence_output], dim=1)
        attention_mask = torch.cat([title_mask, frame_mask], dim=1)
        sequence_output = self.mix_encoder(embedding, attention_mask)[0]
        if task == 'mlm_train':
            mlm_mask = batch['mlm_mask'].cuda()
            mlm_labels = batch['mlm_labels'].cuda()
            mlm_labels = mlm_labels[mlm_mask[:, :32]].contiguous().view(-1)
            masked_output = self._compute_masked_hidden(sequence_output, mlm_mask)
            prediction_scores = self.lm_head(masked_output)
            masked_lm_loss = F.cross_entropy(prediction_scores, mlm_labels, reduction='mean')
            return masked_lm_loss
        elif task == 'mlm_val':
            mlm_mask = batch['mlm_mask'].cuda()
            mlm_labels = batch['mlm_labels'].cuda()
            mlm_labels = mlm_labels[mlm_mask[:, :32]].contiguous().view(-1)
            masked_output = self._compute_masked_hidden(sequence_output, mlm_mask)
            prediction_scores = self.lm_head(masked_output)
            return prediction_scores.cpu(), mlm_labels.cpu()
        else:
            assert "wrong task name"

    def forward_repr(self, batch):
        # embedding layer
        embedding, attention_mask = self._compute_embeddings(batch)
        encoder_outputs = self.encoder(embedding, attention_mask)

        sequence_output = encoder_outputs[0]
        output = sequence_output
        # pooled_output = self.pooler(sequence_output)
        # output = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return output

    def _compute_sep_embeddings(self, batch):
        input_ids=batch['input_ids'].cuda()
        title_mask = batch['title_mask'].long().cuda()
        title_embedding = self.embeddings(input_ids) # 单纯的投射关系 + pos_embed [+ type_embed]
        
        frame_feat = batch['frame_feature'].cuda()
        frame_mask = batch['frame_mask'].long().cuda()
        frame_embedding = self.img_embeddings(frame_feat)
        return title_embedding, title_mask, frame_embedding, frame_mask


    def _compute_masked_hidden(self, hidden, mask):
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked