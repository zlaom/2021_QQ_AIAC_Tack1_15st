import random

import torch
from torch.utils.data import Dataset
import copy


# 注意这里传入的要是单个句子的token列表，而不能是tensor形式
def random_word(tokens, vocab_range, mask, attention_mask, mask_prob=0.15):
    """
    Masking some random tokens for Language Model task with probabilities as in
        the original BERT paper.
    :param tokens: list of int, tokenized sentence.
    :param vocab_range: for choosing a random word
    :return: (list of int, list of int), masked tokens and related labels for
        LM prediction
    """
    output_label = []
    txt_len = attention_mask.count(1)
    for i, token in enumerate(tokens):
        if i >= txt_len:
            output_label.append(-1)
            continue
        prob = random.random()
        # mask token with 15% probability
        if prob < mask_prob:
            prob /= mask_prob

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = mask

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(range(*vocab_range)))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            output_label.append(token)
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
    if all(o == -1 for o in output_label):
        # at least mask 1
        output_label[0] = tokens[0]
        tokens[0] = mask

    return tokens, output_label


def asr_random_word(tokens, vocab_range, mask, attention_mask, mask_prob=0.15):
    """
    Masking some random tokens for Language Model task with probabilities as in
        the original BERT paper.
    :param tokens: list of int, tokenized sentence.
    :param vocab_range: for choosing a random word
    :return: (list of int, list of int), masked tokens and related labels for
        LM prediction
    """
    output_label = []
    txt_len = attention_mask.count(1)
    for i, token in enumerate(tokens):
        if i >= txt_len:
            output_label.append(-1)
            continue
        if i == 0:
            output_label.append(-1)
            continue
        prob = random.random()
        # mask token with 15% probability
        if prob < mask_prob:
            prob /= mask_prob

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = mask

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(range(*vocab_range)))

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            output_label.append(token)
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)
    if all(o == -1 for o in output_label):
        # at least mask 1
        output_label[0] = tokens[0]
        tokens[0] = mask

    return tokens, output_label


def _get_txt_tgt_mask(txt_mask, n_frame):
    z = torch.zeros(n_frame, dtype=torch.bool)
    txt_mask_tgt = torch.cat([z, txt_mask], dim=0)
    return txt_mask_tgt


def create_mlm_io(input_ids, db, mask_prob, cls_tok=True):
    input_ids, txt_labels = random_word(input_ids, db.v_range, db.mask, mask_prob)
    if cls_tok:
        input_ids = [db.cls_] + input_ids
    else:
        input_ids = [db.sep] + input_ids
    txt_labels = torch.tensor([-1] + txt_labels)
    return input_ids, txt_labels


class VideoMlmDataset(Dataset):
    def __init__(self, video_ids, vid_sub_db, mask_prob=0.15, sub_ctx_len=0):
        self.mask_prob = mask_prob
        self.vid_sub_db = vid_sub_db
        self.ids = video_ids
        self.sub_ctx_len = sub_ctx_len

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        vid = self.ids[i]
        example = self.vid_sub_db.txt_db[vid]
        v_feat, nframes = self.vid_sub_db._get_v_feat(vid)
        sub2frames = self.vid_sub_db.vid_sub2frame[vid]
        num_subs = len(sub2frames)
        outputs = []
        for sub_idx, matched_frames in sub2frames:
            # text input
            orig_input_ids = []
            for tmp_sub_idx in range(sub_idx - self.sub_ctx_len, sub_idx + 1):
                if tmp_sub_idx >= 0 and tmp_sub_idx < num_subs:
                    in_ids = example['input_ids'][tmp_sub_idx]
                    if self.vid_sub_db.max_txt_len != -1:
                        in_ids = in_ids[:self.vid_sub_db.max_txt_len]
                    orig_input_ids.extend(copy.deepcopy(in_ids))
            input_ids, txt_labels = create_mlm_io(orig_input_ids, self.vid_sub_db.txt_db,
                                                  self.mask_prob)

            # video input
            n_frame = len(matched_frames)
            if n_frame:
                matched_v_feats = torch.index_select(v_feat, 0, torch.tensor(matched_frames))
                attn_masks = torch.ones(len(input_ids) + n_frame, dtype=torch.long)
                txt_mask_tgt = _get_txt_tgt_mask(txt_labels != -1, n_frame)
            else:
                matched_v_feats = torch.zeros(1, v_feat.shape[1])
                attn_masks = torch.ones(len(input_ids) + 1, dtype=torch.long)
                attn_masks.data[0].fill_(0)
                txt_mask_tgt = _get_txt_tgt_mask(txt_labels != -1, 1)
            input_ids = torch.tensor(input_ids)
            outputs.append((input_ids, matched_v_feats, attn_masks, txt_mask_tgt, txt_labels))

        return outputs
