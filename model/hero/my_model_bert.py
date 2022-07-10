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

class CrossModalTrm(RobertaPreTrainedModel):

    def __init__(self, config, frame_dim=1536, max_frame_seq_len=32):
        super().__init__(config)
        self.encoder = BertEncoder(config)
        self.embeddings = SubEmbeddings(config)

        self.img_embeddings = ImageEmbeddings(config, frame_dim, max_frame_seq_len)
        self.pooler = BertPooler(config)
        self.apply(self.init_weights)
        self.config = config
        self.frame_len = max_frame_seq_len

        self.bert = transformers.BertModel.from_pretrained('pretrain/macbert')
        # pretraining
        self.lm_head = BertLMPredictionHead(config, self.embeddings.word_embeddings.weight)

    def forward(self, batch, task='mlm'):
        if task == 'mlm':
            output = self.forward_mlm(batch)
        elif task == 'repr':
            output = self.forward_repr(batch)
        elif task == 'vsm':
            output = self.forward_vsm(batch) # only title
        else:
            assert "wrong task name"
        return output


    def forward_mlm(self, batch, compute_loss=True):
        embedding, attention_mask = self._compute_embeddings(batch)
        sequence_output = self.encoder(embedding, attention_mask)[0]

        mlm_labels = batch['mlm_labels']
        mlm_title_mask = (mlm_labels != -1)
        B = mlm_title_mask.size(0)
        mlm_total_mask = torch.cat([mlm_title_mask, torch.zeros(B, self.frame_len, dtype=bool)], 1)
        mlm_labels = mlm_labels[mlm_title_mask].contiguous().view(-1)

        mlm_labels = mlm_labels.cuda()
        mlm_title_mask = mlm_title_mask.cuda()
        mlm_total_mask = mlm_total_mask.cuda()


        masked_output = self._compute_masked_hidden(sequence_output, mlm_total_mask)
        prediction_scores = self.lm_head(masked_output)

        return prediction_scores, mlm_labels


    def forward_vsm(self, batch):
        input_ids=batch['input_ids'].cuda()
        title_mask = batch['title_mask'].long().cuda()
        
        # token_type_ids = torch.zeros_like(title_mask).cuda()
        title_embedding = self.bert(input_ids=input_ids, attention_mask=title_mask)[0]
        # title_embedding = self.embeddings(input_ids) # 单纯的投射关系 + pos_embed [+ type_embed]
        encoder_outputs = self.encoder(title_embedding, title_mask)

        sequence_output = encoder_outputs[0]
        output = sequence_output
        return output

    def forward_repr(self, batch):
        # embedding layer
        embedding, attention_mask = self._compute_embeddings(batch)
        encoder_outputs = self.encoder(embedding, attention_mask)

        sequence_output = encoder_outputs[0]
        # pooled_output = self.pooler(sequence_output)
        # output = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # sequence_output, pooled_output, (hidden_states), (attentions)
        return sequence_output

    def _compute_embeddings(self, batch):
        input_ids=batch['input_ids'].cuda()
        title_mask = batch['title_mask'].cuda()

        # token_type_ids = torch.zeros_like(title_mask).cuda()
        title_embedding = self.bert(input_ids=input_ids, attention_mask=title_mask)[0]
        # title_embedding = self.embeddings(input_ids) # 单纯的投射关系 + pos_embed [+ type_embed]
        
        frame_feat = batch['frame_feature'].cuda()
        frame_mask = batch['frame_mask'].cuda()

        # token_type_id
        # device = frame_mask.device
        # frame_type_embeddings = self.embeddings.token_type_embeddings(torch.ones(1, 1, dtype=torch.long, device=device))
        # frame_embedding = self.img_embeddings(frame_feat, frame_type_embeddings) # 单纯的投射关系 + pos_embed [+ type_embed]

        frame_embedding = self.img_embeddings(frame_feat)

        embedding = torch.cat([title_embedding, frame_embedding], dim=1).cuda()
        attention_mask = torch.cat([title_mask.long(), frame_mask.long()], dim=1).cuda()

        return embedding, attention_mask

    def _compute_masked_hidden(self, hidden, mask):
        mask = mask.unsqueeze(-1).expand_as(hidden)
        hidden_masked = hidden[mask].contiguous().view(-1, hidden.size(-1))
        return hidden_masked



class TemporalTrm(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.embeddings = FrameEmbeddings(config) # 位置编码 + layernorm+dropout
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_weights)

    def forward_encoder(self, embedding, attention_mask, pool=False):
        encoder_outputs = self.encoder(embedding, attention_mask)

        sequence_output = encoder_outputs[0]
        if pool: # pool 用作整体特征输出，处理cls token，如果不分类或者没加cls token是用不到的
            pooled_output = self.pooler(sequence_output)
            return pooled_output
        return sequence_output

    def forward(self, frame_feat, attention_mask):
        # embedding layer
        # embedding_output = self.embeddings(frame_feat)
        embedding_output = frame_feat
        output = self.forward_encoder(embedding_output, attention_mask)
        return output




class HierarchicalVlModel(VideoPreTrainedModel):
    def __init__(self, config, vfeat_dim=1536, nce_temp=1.0):
        super().__init__(config)
        self.frame_len = 32
        self.f_encoder = CrossModalTrm(config.f_config, vfeat_dim)
        self.frame_transform = LinearLayer(vfeat_dim, config.f_config.hidden_size,
            layer_norm=True, dropout=config.f_config.hidden_dropout_prob, relu=True)
        self.c_encoder = TemporalTrm(config.c_config)

        # mfm used
        self.feat_regress = FrameFeatureRegression(config.f_config.hidden_size, vfeat_dim) # 768->1536
        self.nce_temp = nce_temp  # NCE shares same head with regression
        self.mask_embedding = nn.Embedding(2, vfeat_dim, padding_idx=0)
        
        # vsm used
        self.qfeat_dim = config.f_config.hidden_size
        self.query_encoder = QueryFeatEncoder(config.q_config, self.qfeat_dim)
        self.neg_ctx = 0.5
        self.neg_q = 0.5
        self.use_all_neg = True
        self.ranking_loss_type = 'hinge'
        self.use_hard_negative = False
        self.hard_pool_size = 20 # 由于上面是False，这两个参数应该没用到
        self.hard_neg_weight = 10
        self.margin = 0.1

        # tag used
        self.tag_loss = torch.nn.BCEWithLogitsLoss()
        self.cls = TagLinear(config.f_config.hidden_size)

    def forward(self, batch, task, compute_loss=True):
        if task == 'mlm':
            output = self.forward_mlm(batch, compute_loss)
        elif task == 'mfm':
            output = self.forward_mfm(batch, compute_loss)
        elif task == 'vsm':
            output = self.forward_vsm(batch, compute_loss)
        elif task == 'tag':
            output = self.forward_tag(batch, compute_loss)
        elif task == 'repr':
            output = self.f_encoder.forward_repr(batch)
        else:
            assert "wrong task name"
        return output

    def forward_tag(self, batch, compute_loss=True):
        sequence_outputs = self.f_encoder(batch, 'repr')
        cls_token = sequence_outputs[:, 0]
        if compute_loss:
            tag_ids = batch['tag_id'].float().cuda()
            prediction_score = self.cls(cls_token)
            loss = self.tag_loss(prediction_score, tag_ids)
            return loss
        else:
            normed_embedding = F.normalize(cls_token, p=2, dim=1)
            return normed_embedding

    def forward_mlm(self, batch, compute_loss=True):
        prediction_scores, mlm_labels = self.f_encoder(batch, 'mlm')
        if compute_loss:
            masked_lm_loss = F.cross_entropy(prediction_scores, mlm_labels, reduction='mean')
            return masked_lm_loss
        else:
            return prediction_scores.cpu(), mlm_labels.cpu()

    def forward_vsm(self, batch, compute_loss):
        # 融合后的视频信息
        sequence_outputs = self.f_encoder(batch, 'repr')
        frame_embeddings = sequence_outputs[:, -self.frame_len:] # 取视频帧，后32帧
        # query
        query_sequence_output = self.f_encoder(batch, 'vsm')
        modularized_query = self.query_encoder(query_sequence_output, batch['title_mask'].cuda())

        q2video_scores = self.get_video_level_scores(modularized_query, frame_embeddings, batch['frame_mask'].cuda())
        reduction = 'mean' if compute_loss else 'sum' # 这里的compute_loss 只是用来区分train和val
        loss_neg_ctx, loss_neg_q = self.get_video_level_loss(q2video_scores, reduction)
        loss_neg_ctx = self.neg_ctx * loss_neg_ctx
        loss_neg_q = self.neg_q * loss_neg_q
        loss = loss_neg_ctx + loss_neg_q
        if compute_loss == True: # 这里的compute_loss 只是用来区分train和val
            return loss
        else:
            return loss_neg_ctx, loss_neg_q

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
        mask = self.mask_embedding(mfm_mask.long()) # 加mask embedding
        frame_feat = frame_feat + mask
        batch['frame_feature'] = frame_feat

        # 开始通过模型
        sequence_outputs = self.f_encoder(batch, 'repr')
        frame_sequence_outputs = sequence_outputs[:, -self.frame_len:] # 取视频帧，后32帧
        # attention_mask = batch['frame_mask'].long().cuda()
        # residual = self.frame_transform(frame_feat)
        # frame_sequence_outputs = frame_sequence_outputs + residual
        # frame_sequence_outputs = self.c_encoder(frame_sequence_outputs, attention_mask)
        

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


    def get_video_level_scores(self, modularized_query,
                               context_feat1, context_mask):
        """ Calculate video2query scores for each pair of video
            and query inside the batch.
        Args:
            modularized_query: (N, D)
            context_feat1: (N, L, D),
                output of the first transformer encoder layer
            context_mask: (N, L)
        Returns:
            context_query_scores: (N, N)
                score of each query w.r.t. each video inside the batch,
                diagonal positions are positive. used to get negative samples.
        """
        modularized_query = F.normalize(modularized_query, dim=-1, eps=1e-5)
        context_feat1 = F.normalize(context_feat1, dim=-1, eps=1e-5)

        query_context_scores = torch.einsum(
            "md,nld->mln", modularized_query, context_feat1)  # (N, L, N)
        context_mask = context_mask.transpose(0, 1).unsqueeze(0)  # (1, L, N)
        context_mask = context_mask.to(dtype=query_context_scores.dtype)  # fp16 compatibility
        query_context_scores = mask_logits( query_context_scores, context_mask)  # (N, L, N)
        query_context_scores, _ = torch.max(query_context_scores, dim=1)  # (N, N) diagonal positions are positive pairs.
        return query_context_scores

    def get_video_level_loss(self, query_context_scores, reduction="mean"):
        """ ranking loss between (pos. query + pos. video)
            and (pos. query + neg. video) or (neg. query + pos. video)
        Args:
            query_context_scores: (Nq, Nv), cosine similarity [-1, 1],
                Each row contains the scores
                between the query to each of the videos inside the batch.
        """
        bsz_q, bsz_v = query_context_scores.size()  # (Nq, Nv)
        num_q_per_v = int(bsz_q/bsz_v)
        loss_neg_ctx = torch.tensor(0).to(query_context_scores.device)
        loss_neg_q = torch.tensor(0).to(query_context_scores.device)
        if bsz_v == 1:
            return loss_neg_ctx, loss_neg_q

        # (Nq, Nv)
        query_context_scores_masked = query_context_scores.clone()
        pos_video_query_scores = []
        for i, j in zip(range(bsz_v), range(0, bsz_q, num_q_per_v)):
            pos_video_query_scores.append(
                query_context_scores[j: j+num_q_per_v, i])
            # impossibly large for cosine similarity, the copy is created
            # as modifying the original will cause error
            query_context_scores_masked[
                j: j+num_q_per_v, i] = 999
        # (Nv, 5)
        pos_video_query_scores = torch.stack(pos_video_query_scores, dim=0)
        video_query_scores_masked = query_context_scores_masked.transpose(0, 1)
        # (Nq, 1)
        pos_query_video_scores = pos_video_query_scores.view(bsz_q, 1)

        if self.use_all_neg:
            # get negative videos per query
            # (Nq, Nv-1)
            pos_query_neg_context_scores = self.get_all_neg_scores(
                query_context_scores_masked,
                sample_min_idx=1)
            # (Nq, Nv-1)
            loss_neg_ctx = self.get_ranking_loss(
                pos_query_video_scores, pos_query_neg_context_scores)
            if self.use_hard_negative:
                weighting_mat = torch.ones_like(loss_neg_ctx)
                weighting_mat[:, self.hard_pool_size:] = .1
                weighting_mat[:, :self.hard_pool_size] = self.hard_neg_weight
                loss_neg_ctx = weighting_mat * loss_neg_ctx

            # get negative query per video
            # (Nv, Nq-5)
            neg_query_pos_context_scores = self.get_all_neg_scores(
                video_query_scores_masked,
                sample_min_idx=num_q_per_v)
            # (Nv, 1, Nq-5)
            neg_query_pos_context_scores =\
                neg_query_pos_context_scores.unsqueeze(1)
            # (Nv, 5, 1)
            pos_video_query_scores = pos_video_query_scores.unsqueeze(-1)
            # (Nv, 5, Nq-5)
            loss_neg_q = self.get_ranking_loss(
                pos_video_query_scores, neg_query_pos_context_scores)
            # (Nq, Nq-5)
            loss_neg_q = loss_neg_q.view(-1, loss_neg_q.size(2))
            if self.use_hard_negative:
                weighting_mat = torch.ones_like(loss_neg_q)
                weighting_mat[:, self.hard_pool_size:] = .1
                weighting_mat[:, :self.hard_pool_size] = self.hard_neg_weight
                loss_neg_q = weighting_mat * loss_neg_q
        else:
            # (Nq, 1)
            pos_query_neg_context_scores = self.get_sampled_neg_scores(
                query_context_scores_masked,
                sample_min_idx=1).unsqueeze(-1)
            # (Nq, 1)
            loss_neg_ctx = self.get_ranking_loss(
                pos_query_video_scores, pos_query_neg_context_scores)
            # (Nv, 1)
            neg_query_pos_context_scores = self.get_sampled_neg_scores(
                video_query_scores_masked,
                sample_min_idx=num_q_per_v).unsqueeze(-1)
            # (Nv, 5)
            loss_neg_q = self.get_ranking_loss(
                pos_video_query_scores, neg_query_pos_context_scores)

        if reduction == "sum":
            return loss_neg_ctx.mean(1), loss_neg_q.mean(1)
        elif reduction == "mean":
            return loss_neg_ctx.mean(1).mean(0), loss_neg_q.mean(1).mean(0)
        elif reduction is None:
            return loss_neg_ctx, loss_neg_q
        else:
            raise NotImplementedError(f"reduction {reduction} not supported")

    def get_all_neg_scores(self, scores_masked,
                           pos_indices=None, sample_min_idx=1):
        """
        scores: (N, N), cosine similarity [-1, 1],
            Each row are scores: query --> all videos.
            Transposed version: video --> all queries.
        scores_masked: (N, N) the same as scores,
            except that the diagonal (positive) positions
            are masked with a large value.
        """
        bsz, sample_size = scores_masked.size()
        assert sample_size > sample_min_idx,\
            "Unable to sample negative when bsz==sample_min_idx"
        if pos_indices is None:
            pos_indices = torch.arange(bsz).to(scores_masked.device)

        sorted_scores_masked, sorted_scores_indices = torch.sort(
            scores_masked, descending=True, dim=1)
        # skip the masked positive
        # (N, sample_size-sample_min_idx)
        neg_scores = sorted_scores_masked[:, sample_min_idx:]
        return neg_scores

    def get_ranking_loss(self, pos_score, neg_score):
        """ Note here we encourage positive scores to be larger
            than negative scores.
        Args:
            pos_score: (N, ), torch.float32
            neg_score: (N, ), torch.float32
        """
        if self.ranking_loss_type == "hinge":
            # max(0, m + S_neg - S_pos)
            loss = torch.clamp(
                self.margin + neg_score - pos_score,
                min=0)
        elif self.ranking_loss_type == "lse":
            # log[1 + exp(S_neg - S_pos)]
            loss = torch.log1p(
                torch.exp(neg_score - pos_score))
        else:
            raise NotImplementedError("Only support 'hinge' and 'lse'")

        return loss

    def get_sampled_neg_scores(self, scores_masked, sample_min_idx=1):
        """
        scores_masked: (Nq, Nv)
            except that the diagonal (positive) positions
            are masked with a large value.
        """
        bsz, sample_size = scores_masked.size()
        assert sample_size > sample_min_idx,\
            "Unable to sample negative when bsz==sample_min_idx"
        num_neg = bsz
        pos_indices = torch.arange(bsz).to(scores_masked.device)

        _, sorted_scores_indices = torch.sort(
            scores_masked, descending=True, dim=1)
        # skip the masked positive
        sample_max_idx = min(
            sample_min_idx + self.hard_pool_size, sample_size) \
            if self.use_hard_negative else sample_size
        sampled_neg_score_indices = sorted_scores_indices[
            pos_indices, torch.randint(
                sample_min_idx, sample_max_idx,
                size=(num_neg,)).to(scores_masked.device)]  # (N, )
        sampled_neg_scores = scores_masked[
            pos_indices, sampled_neg_score_indices]  # (N, )
        return sampled_neg_scores