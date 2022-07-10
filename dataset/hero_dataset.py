from google.protobuf.descriptor import EnumValueDescriptor
import torch
import json
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
from dataset.reader_base import read_single_record_with_spec_index, prese_row, get_index_by_id, get_index_by_index

# mlm
import transformers
from dataset.hero.mlm import random_word, asr_random_word


class TagValDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, id_list, data_info, test=False):
        self.cfg = cfg
        self.id_list = id_list
        self.data_info = data_info
        self.test = test
        desc_path = cfg["DESC_PATH"]
        with open(desc_path) as fh:
            self.desc = json.load(fh)

        tag_count = pd.read_csv(cfg['TAG_COUNT_INFO'])
        tag_count = tag_count.sort_values(by='count', ascending=False)
        self.selected_tags = set(tag_count.iloc[:cfg['TAG_SIZE']]['tag_id'])
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([self.selected_tags])

        category_id = pd.read_csv(cfg['CATEGORY_ID_COUNT_INFO'])
        category_count = category_id.sort_values(by='count', ascending=False)
        self.category_id = np.array(category_count.iloc[:cfg['CATEGORY_SIZE']]['category_id'])
        self.ohc = OneHotEncoder()
        self.ohc.fit(self.category_id.reshape(-1, 1))

        # mlm
        self.CLS = 102
        self.tokenizer = transformers.BertTokenizer.from_pretrained('data/chinese_macbert_base')

    def transform(self, record):
        # 筛选的tag
        tags = [t for t in record['tag_id'] if t in self.selected_tags]
        multi_hot = self.mlb.transform([tags])[0]
        record['tag_id'] = multi_hot

    def __getitem__(self, index):
        id = self.id_list[index]
        _, tfrecord_data_file, start_offset, end_offset = get_index_by_id(self.data_info, id)
        row = read_single_record_with_spec_index(tfrecord_data_file, start_offset, end_offset,
                                                 self.desc)
        _, record = prese_row(row, self.desc)
        if not self.test:
            self.transform(record)
        else:
            if 'tag_id' in record.keys():
                del record['tag_id']
            if 'category_id' in record.keys():
                del record['category_id']
        # tokenize
        tokenized = self.tokenizer(record['title'],
                                   add_special_tokens=False,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=32)
        input_ids = tokenized['input_ids']
        title_mask = tokenized['attention_mask']
        input_ids = np.array([self.CLS] + input_ids)
        title_mask = np.array([1] + title_mask)
        # input_ids = np.array(input_ids)
        # title_mask = np.array(title_mask)
        record['only_input_ids'] = input_ids
        record['only_title_mask'] = title_mask

        return record

    def __len__(self):
        return len(self.id_list)


class AsrTagDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, id_list, data_info, test=False):
        self.cfg = cfg
        self.id_list = id_list
        self.data_info = data_info
        self.test = test
        desc_path = cfg["DESC_PATH"]
        with open(desc_path) as fh:
            self.desc = json.load(fh)

        tag_count = pd.read_csv(cfg['TAG_COUNT_INFO'])
        tag_count = tag_count.sort_values(by='count', ascending=False)
        self.selected_tags = set(tag_count.iloc[:cfg['TAG_SIZE']]['tag_id'])
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([self.selected_tags])

        category_id = pd.read_csv(cfg['CATEGORY_ID_COUNT_INFO'])
        category_count = category_id.sort_values(by='count', ascending=False)
        self.category_id = np.array(category_count.iloc[:cfg['CATEGORY_SIZE']]['category_id'])
        self.ohc = OneHotEncoder()
        self.ohc.fit(self.category_id.reshape(-1, 1))

        # mlm
        self.CLS = 102
        self.tokenizer = transformers.BertTokenizer.from_pretrained('data/chinese_macbert_base')

    def transform(self, record):
        # 筛选的tag
        tags = [t for t in record['tag_id'] if t in self.selected_tags]
        multi_hot = self.mlb.transform([tags])[0]
        record['tag_id'] = multi_hot

    def __getitem__(self, index):
        id = self.id_list[index]
        _, tfrecord_data_file, start_offset, end_offset = get_index_by_id(self.data_info, id)
        row = read_single_record_with_spec_index(tfrecord_data_file, start_offset, end_offset,
                                                 self.desc)
        _, record = prese_row(row, self.desc)
        if not self.test:
            self.transform(record)
        else:
            if 'tag_id' in record.keys():
                del record['tag_id']
            if 'category_id' in record.keys():
                del record['category_id']
        # tokenize
        tokenized = self.tokenizer([record['title_asr']],
                                   truncation=True,
                                   padding='max_length',
                                   max_length=64)
        input_ids = tokenized['input_ids'][0]
        token_type_ids = tokenized['token_type_ids'][0]
        title_mask = tokenized['attention_mask'][0]

        input_ids = np.array(input_ids)
        title_mask = np.array(title_mask)
        token_type_ids = np.array(token_type_ids)

        record['input_ids'] = input_ids
        record['token_type_ids'] = token_type_ids
        record['title_mask'] = title_mask
        return record

    def __len__(self):
        return len(self.id_list)


class LargeAsrTagDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, id_list, data_info, test=False):
        self.cfg = cfg
        self.id_list = id_list
        self.data_info = data_info
        self.test = test
        desc_path = cfg["DESC_PATH"]
        with open(desc_path) as fh:
            self.desc = json.load(fh)

        tag_count = pd.read_csv(cfg['TAG_COUNT_INFO'])
        tag_count = tag_count.sort_values(by='count', ascending=False)
        self.selected_tags = set(tag_count.iloc[:cfg['TAG_SIZE']]['tag_id'])
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([self.selected_tags])

        category_id = pd.read_csv(cfg['CATEGORY_ID_COUNT_INFO'])
        category_count = category_id.sort_values(by='count', ascending=False)
        self.category_id = np.array(category_count.iloc[:cfg['CATEGORY_SIZE']]['category_id'])
        self.ohc = OneHotEncoder()
        self.ohc.fit(self.category_id.reshape(-1, 1))

        # mlm
        self.CLS = 102
        self.tokenizer = transformers.BertTokenizer.from_pretrained('hfl/chinese-macbert-large')

    def transform(self, record):
        # 筛选的tag
        tags = [t for t in record['tag_id'] if t in self.selected_tags]
        multi_hot = self.mlb.transform([tags])[0]
        record['tag_id'] = multi_hot

    def __getitem__(self, index):
        id = self.id_list[index]
        _, tfrecord_data_file, start_offset, end_offset = get_index_by_id(self.data_info, id)
        row = read_single_record_with_spec_index(tfrecord_data_file, start_offset, end_offset,
                                                 self.desc)
        _, record = prese_row(row, self.desc)
        if not self.test:
            self.transform(record)
        else:
            if 'tag_id' in record.keys():
                del record['tag_id']
            if 'category_id' in record.keys():
                del record['category_id']
        # tokenize
        tokenized = self.tokenizer([record['title_asr']],
                                   truncation=True,
                                   padding='max_length',
                                   max_length=64)
        input_ids = tokenized['input_ids'][0]
        token_type_ids = tokenized['token_type_ids'][0]
        title_mask = tokenized['attention_mask'][0]

        input_ids = np.array(input_ids)
        title_mask = np.array(title_mask)
        token_type_ids = np.array(token_type_ids)

        record['input_ids'] = input_ids
        record['token_type_ids'] = token_type_ids
        record['title_mask'] = title_mask
        return record

    def __len__(self):
        return len(self.id_list)


class AsrVsmDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, index_list, data_info):
        self.cfg = cfg
        self.index_list = index_list
        self.data_info = data_info
        desc_path = cfg["DESC_PATH"]
        with open(desc_path) as fh:
            self.desc = json.load(fh)

        tag_count = pd.read_csv(cfg['TAG_COUNT_INFO'])
        tag_count = tag_count.sort_values(by='count', ascending=False)
        self.selected_tags = set(tag_count.iloc[:cfg['TAG_SIZE']]['tag_id'])
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([self.selected_tags])

        # mlm
        self.CLS = 102
        self.tokenizer = transformers.BertTokenizer.from_pretrained('data/chinese_macbert_base')

    def transform(self, record, num_frame):
        # tag_id
        tags = [t for t in record['tag_id'] if t in self.selected_tags]
        multi_hot = self.mlb.transform([tags])[0]
        record['tag_id'] = multi_hot

        tokenized = self.tokenizer([record['title_asr']],
                                   truncation=True,
                                   padding='max_length',
                                   max_length=64)
        input_ids = tokenized['input_ids'][0]
        token_type_ids = tokenized['token_type_ids'][0]
        title_mask = tokenized['attention_mask'][0]

        input_ids = np.array(input_ids)
        title_mask = np.array(title_mask)
        token_type_ids = np.array(token_type_ids)

        record['input_ids'] = input_ids
        record['token_type_ids'] = token_type_ids
        record['title_mask'] = title_mask
        return record

    def __getitem__(self, index):
        index = self.index_list[index]
        _, tfrecord_data_file, start_offset, end_offset = get_index_by_index(self.data_info, index)
        row = read_single_record_with_spec_index(tfrecord_data_file, start_offset, end_offset,
                                                 self.desc)
        num_frame, record = prese_row(row, self.desc)
        self.transform(record, num_frame)
        return record

    def __len__(self):
        return len(self.index_list)


class LargeAsrVsmDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, index_list, data_info):
        self.cfg = cfg
        self.index_list = index_list
        self.data_info = data_info
        desc_path = cfg["DESC_PATH"]
        with open(desc_path) as fh:
            self.desc = json.load(fh)

        tag_count = pd.read_csv(cfg['TAG_COUNT_INFO'])
        tag_count = tag_count.sort_values(by='count', ascending=False)
        self.selected_tags = set(tag_count.iloc[:cfg['TAG_SIZE']]['tag_id'])
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([self.selected_tags])

        # mlm
        self.CLS = 102
        self.tokenizer = transformers.BertTokenizer.from_pretrained('hfl/chinese-macbert-large')

    def transform(self, record, num_frame):
        # tag_id
        tags = [t for t in record['tag_id'] if t in self.selected_tags]
        multi_hot = self.mlb.transform([tags])[0]
        record['tag_id'] = multi_hot

        tokenized = self.tokenizer([record['title_asr']],
                                   truncation=True,
                                   padding='max_length',
                                   max_length=64)
        input_ids = tokenized['input_ids'][0]
        token_type_ids = tokenized['token_type_ids'][0]
        title_mask = tokenized['attention_mask'][0]

        input_ids = np.array(input_ids)
        title_mask = np.array(title_mask)
        token_type_ids = np.array(token_type_ids)

        record['input_ids'] = input_ids
        record['token_type_ids'] = token_type_ids
        record['title_mask'] = title_mask
        return record

    def __getitem__(self, index):
        index = self.index_list[index]
        _, tfrecord_data_file, start_offset, end_offset = get_index_by_index(self.data_info, index)
        row = read_single_record_with_spec_index(tfrecord_data_file, start_offset, end_offset,
                                                 self.desc)
        num_frame, record = prese_row(row, self.desc)
        self.transform(record, num_frame)
        return record

    def __len__(self):
        return len(self.index_list)


class VsmDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, index_list, data_info):
        self.cfg = cfg
        self.index_list = index_list
        self.data_info = data_info
        desc_path = cfg["DESC_PATH"]
        with open(desc_path) as fh:
            self.desc = json.load(fh)

        tag_count = pd.read_csv(cfg['TAG_COUNT_INFO'])
        tag_count = tag_count.sort_values(by='count', ascending=False)
        self.selected_tags = set(tag_count.iloc[:cfg['TAG_SIZE']]['tag_id'])
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([self.selected_tags])

        # mlm
        self.CLS = 102
        self.tokenizer = transformers.BertTokenizer.from_pretrained('data/chinese_macbert_base')

    def transform(self, record, num_frame):
        # tag_id
        tags = [t for t in record['tag_id'] if t in self.selected_tags]
        multi_hot = self.mlb.transform([tags])[0]
        record['tag_id'] = multi_hot

        # tokenize
        tokenized = self.tokenizer(record['title'],
                                   add_special_tokens=False,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=32)
        input_ids = tokenized['input_ids']
        title_mask = tokenized['attention_mask']
        input_ids = np.array([self.CLS] + input_ids)
        title_mask = np.array([1] + title_mask)
        # input_ids = np.array(input_ids)
        # title_mask = np.array(title_mask)
        record['only_input_ids'] = input_ids
        record['only_title_mask'] = title_mask

        return record

    def __getitem__(self, index):
        index = self.index_list[index]
        _, tfrecord_data_file, start_offset, end_offset = get_index_by_index(self.data_info, index)
        row = read_single_record_with_spec_index(tfrecord_data_file, start_offset, end_offset,
                                                 self.desc)
        num_frame, record = prese_row(row, self.desc)
        self.transform(record, num_frame)
        return record

    def __len__(self):
        return len(self.index_list)


class MfmDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, index_list, data_info):
        self.cfg = cfg
        self.index_list = index_list
        self.data_info = data_info
        desc_path = cfg["DESC_PATH"]
        with open(desc_path) as fh:
            self.desc = json.load(fh)

        tag_count = pd.read_csv(cfg['TAG_COUNT_INFO'])
        tag_count = tag_count.sort_values(by='count', ascending=False)
        self.selected_tags = set(tag_count.iloc[:cfg['TAG_SIZE']]['tag_id'])
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([self.selected_tags])

        # mlm
        self.CLS = 102
        self.tokenizer = transformers.BertTokenizer.from_pretrained('data/chinese_macbert_base')

    # mfm 决定mask的帧，不mask pad帧
    def _get_img_mask(self, mask_prob, num_frame, max_frame=32):
        img_mask = [random.random() < mask_prob for _ in range(num_frame)] + \
                [False for _ in range(max_frame - num_frame)]
        if not any(img_mask):
            img_mask[random.choice(range(num_frame))] = True
        if num_frame == 1:
            img_mask[0] = False
        img_mask = torch.tensor(img_mask)
        return img_mask

    def transform(self, record, num_frame):
        # 原来的几句，不知道为啥删了会报错
        tags = [t for t in record['tag_id'] if t in self.selected_tags]
        multi_hot = self.mlb.transform([tags])[0]
        record['tag_id'] = multi_hot

        # tokenize
        tokenized = self.tokenizer(record['title'],
                                   add_special_tokens=False,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=32)
        input_ids = tokenized['input_ids']
        title_mask = tokenized['attention_mask']
        input_ids = np.array([self.CLS] + input_ids)
        title_mask = np.array([1] + title_mask)
        # input_ids = np.array(input_ids)
        # title_mask = np.array(title_mask)
        record['input_ids'] = input_ids
        record['title_mask'] = title_mask

        # mfm
        mfm_mask = self._get_img_mask(0.15, num_frame)
        record['mfm_mask'] = mfm_mask

        return record

    def __getitem__(self, index):
        index = self.index_list[index]
        _, tfrecord_data_file, start_offset, end_offset = get_index_by_index(self.data_info, index)
        row = read_single_record_with_spec_index(tfrecord_data_file, start_offset, end_offset,
                                                 self.desc)
        num_frame, record = prese_row(row, self.desc)
        self.transform(record, num_frame)
        return record

    def __len__(self):
        return len(self.index_list)


class MlmDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, index_list, data_info):
        self.cfg = cfg
        self.index_list = index_list
        self.data_info = data_info
        desc_path = cfg["DESC_PATH"]
        with open(desc_path) as fh:
            self.desc = json.load(fh)

        tag_count = pd.read_csv(cfg['TAG_COUNT_INFO'])
        tag_count = tag_count.sort_values(by='count', ascending=False)
        self.selected_tags = set(tag_count.iloc[:cfg['TAG_SIZE']]['tag_id'])
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([self.selected_tags])

        # mlm
        self.CLS = 102
        self.tokenizer = transformers.BertTokenizer.from_pretrained('data/chinese_macbert_base')

    def transform(self, record):
        # 筛选的tag
        tags = [t for t in record['tag_id'] if t in self.selected_tags]
        multi_hot = self.mlb.transform([tags])[0]
        record['tag_id'] = multi_hot

        tokenized = self.tokenizer(record['title'],
                                   add_special_tokens=False,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=32)
        input_ids = tokenized['input_ids']
        # token_type_ids = tokenized['token_type_ids']
        title_mask = tokenized['attention_mask']
        input_ids, mlm_labels = random_word(input_ids, [106, 21128],
                                            mask=104,
                                            attention_mask=title_mask,
                                            mask_prob=0.15)
        # add cls token
        input_ids = np.array([self.CLS] + input_ids)
        title_mask = np.array([1] + title_mask)
        mlm_labels = np.array([-1] + mlm_labels)
        # input_ids = np.array(input_ids)
        # title_mask = np.array(title_mask)
        # mlm_labels = np.array(mlm_labels)

        record['input_ids'] = input_ids
        # record['token_type_ids'] = token_type_ids
        record['title_mask'] = title_mask
        record['mlm_labels'] = mlm_labels
        return record

    def __getitem__(self, index):
        index = self.index_list[index]
        _, tfrecord_data_file, start_offset, end_offset = get_index_by_index(self.data_info, index)
        row = read_single_record_with_spec_index(tfrecord_data_file, start_offset, end_offset,
                                                 self.desc)
        _, record = prese_row(row, self.desc)
        self.transform(record)
        return record

    def __len__(self):
        return len(self.index_list)


class AsrMlmDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, index_list, data_info):
        self.cfg = cfg
        self.index_list = index_list
        self.data_info = data_info
        desc_path = cfg["DESC_PATH"]
        with open(desc_path) as fh:
            self.desc = json.load(fh)

        tag_count = pd.read_csv(cfg['TAG_COUNT_INFO'])
        tag_count = tag_count.sort_values(by='count', ascending=False)
        self.selected_tags = set(tag_count.iloc[:cfg['TAG_SIZE']]['tag_id'])
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([self.selected_tags])

        # mlm
        self.CLS = 102
        self.tokenizer = transformers.BertTokenizer.from_pretrained('data/chinese_macbert_base')

    def transform(self, record):
        # 筛选的tag
        tags = [t for t in record['tag_id'] if t in self.selected_tags]
        multi_hot = self.mlb.transform([tags])[0]
        record['tag_id'] = multi_hot

        tokenized = self.tokenizer([record['title_asr']],
                                   truncation=True,
                                   padding='max_length',
                                   max_length=64)
        input_ids = tokenized['input_ids'][0]
        token_type_ids = tokenized['token_type_ids'][0]
        title_mask = tokenized['attention_mask'][0]
        input_ids, mlm_labels = asr_random_word(input_ids, [106, 21128],
                                                mask=104,
                                                attention_mask=title_mask,
                                                mask_prob=0.15)
        # add cls token
        # input_ids = np.array([self.CLS] + input_ids)
        # title_mask = np.array([1] + title_mask)
        # mlm_labels = np.array([-1] + mlm_labels)
        input_ids = np.array(input_ids)
        title_mask = np.array(title_mask)
        token_type_ids = np.array(token_type_ids)
        mlm_labels = np.array(mlm_labels)

        record['input_ids'] = input_ids
        record['token_type_ids'] = token_type_ids
        record['title_mask'] = title_mask
        record['mlm_labels'] = mlm_labels
        return record

    def __getitem__(self, index):
        index = self.index_list[index]
        _, tfrecord_data_file, start_offset, end_offset = get_index_by_index(self.data_info, index)
        row = read_single_record_with_spec_index(tfrecord_data_file, start_offset, end_offset,
                                                 self.desc)
        _, record = prese_row(row, self.desc)
        self.transform(record)
        return record

    def __len__(self):
        return len(self.index_list)


# for spearman
class MlmTestDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, id_list, data_info):
        self.cfg = cfg
        self.id_list = id_list
        self.data_info = data_info
        desc_path = cfg["DESC_PATH"]
        with open(desc_path) as fh:
            self.desc = json.load(fh)

        tag_count = pd.read_csv(cfg['TAG_COUNT_INFO'])
        tag_count = tag_count.sort_values(by='count', ascending=False)
        self.selected_tags = set(tag_count.iloc[:cfg['TAG_SIZE']]['tag_id'])
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([self.selected_tags])

        # mlm
        self.CLS = 102
        self.tokenizer = transformers.BertTokenizer.from_pretrained('data/chinese_macbert_base')

    def transform(self, record):
        # 筛选的tag
        tags = [t for t in record['tag_id'] if t in self.selected_tags]
        multi_hot = self.mlb.transform([tags])[0]
        record['tag_id'] = multi_hot

        tokenized = self.tokenizer(record['title'],
                                   add_special_tokens=False,
                                   truncation=True,
                                   padding='max_length',
                                   max_length=32)
        input_ids = tokenized['input_ids']
        # token_type_ids = tokenized['token_type_ids']
        title_mask = tokenized['attention_mask']
        # input_ids, mlm_labels = random_word(input_ids, [107, 21129], mask=104, attention_mask=title_mask, mask_prob=0.15)
        # add cls token
        input_ids = np.array([self.CLS] + input_ids)
        title_mask = np.array([1] + title_mask)
        # mlm_labels = np.array([-1] + mlm_labels)

        # mlm_mask
        # mlm_mask = np.concatenate([mlm_labels!=-1, np.zeros(32, dtype=np.bool)], 0) # 32帧视频

        record['input_ids'] = input_ids
        # record['token_type_ids'] = token_type_ids
        record['title_mask'] = title_mask
        # record['mlm_mask'] = mlm_mask
        # record['mlm_labels'] = mlm_labels
        return record

    def __getitem__(self, index):
        id = self.id_list[index]
        _, tfrecord_data_file, start_offset, end_offset = get_index_by_id(self.data_info, id)
        row = read_single_record_with_spec_index(tfrecord_data_file, start_offset, end_offset,
                                                 self.desc)
        _, record = prese_row(row, self.desc)
        self.transform(record)
        return record

    def __len__(self):
        return len(self.id_list)


class IndexDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, index_list, data_info):
        self.cfg = cfg
        self.index_list = index_list
        self.data_info = data_info
        desc_path = cfg["DESC_PATH"]
        with open(desc_path) as fh:
            self.desc = json.load(fh)

        tag_count = pd.read_csv(cfg['TAG_COUNT_INFO'])
        tag_count = tag_count.sort_values(by='count', ascending=False)
        self.selected_tags = set(tag_count.iloc[:cfg['TAG_SIZE']]['tag_id'])
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([self.selected_tags])

        category_id = pd.read_csv(cfg['CATEGORY_ID_COUNT_INFO'])
        category_count = category_id.sort_values(by='count', ascending=False)
        self.category_id = np.array(category_count.iloc[:cfg['CATEGORY_SIZE']]['category_id'])
        self.ohc = OneHotEncoder()
        self.ohc.fit(self.category_id.reshape(-1, 1))

    def transform(self, record):
        # tokenizer title

        # 筛选的tag
        tags = [t for t in record['tag_id'] if t in self.selected_tags]
        multi_hot = self.mlb.transform([tags])[0]
        record['tag_id'] = multi_hot

        # 编码category_id
        one_hot = self.ohc.transform([record['category_id']]).toarray()[0]
        # record['category_id'] = one_hot
        record['category_id'] = np.where(one_hot == 1)[0][0]
        return record

    def __getitem__(self, index):
        index = self.index_list[index]
        _, tfrecord_data_file, start_offset, end_offset = get_index_by_index(self.data_info, index)
        row = read_single_record_with_spec_index(tfrecord_data_file, start_offset, end_offset,
                                                 self.desc)
        _, record = prese_row(row, self.desc)
        self.transform(record)
        return record

    def __len__(self):
        return len(self.index_list)


class IdDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, id_list, data_info, test=False):
        self.cfg = cfg
        self.id_list = id_list
        self.data_info = data_info
        self.test = test
        desc_path = cfg["DESC_PATH"]
        with open(desc_path) as fh:
            self.desc = json.load(fh)

        tag_count = pd.read_csv(cfg['TAG_COUNT_INFO'])
        tag_count = tag_count.sort_values(by='count', ascending=False)
        self.selected_tags = set(tag_count.iloc[:cfg['TAG_SIZE']]['tag_id'])
        self.mlb = MultiLabelBinarizer()
        self.mlb.fit([self.selected_tags])

        category_id = pd.read_csv(cfg['CATEGORY_ID_COUNT_INFO'])
        category_count = category_id.sort_values(by='count', ascending=False)
        self.category_id = np.array(category_count.iloc[:cfg['CATEGORY_SIZE']]['category_id'])
        self.ohc = OneHotEncoder()
        self.ohc.fit(self.category_id.reshape(-1, 1))

    def transform(self, record):
        # tokenizer title

        # 筛选的tag
        tags = [t for t in record['tag_id'] if t in self.selected_tags]
        multi_hot = self.mlb.transform([tags])[0]
        record['tag_id'] = multi_hot

        # 编码category_id
        one_hot = self.ohc.transform([record['category_id']]).toarray()[0]
        record['category_id'] = one_hot
        return record

    def __getitem__(self, index):
        id = self.id_list[index]
        _, tfrecord_data_file, start_offset, end_offset = get_index_by_id(self.data_info, id)
        row = read_single_record_with_spec_index(tfrecord_data_file, start_offset, end_offset,
                                                 self.desc)
        _, record = prese_row(row, self.desc)
        if not self.test:
            self.transform(record)
        else:
            if 'tag_id' in record.keys():
                del record['tag_id']
            if 'category_id' in record.keys():
                del record['category_id']

        return record

    def __len__(self):
        return len(self.id_list)


class PairwiseDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, tsv_path, data_info):
        self.cfg = cfg
        self.id_list = []
        self.data_info = data_info
        desc_path = cfg["DESC_PATH"]
        with open(desc_path) as fh:
            self.desc = json.load(fh)

        with open(tsv_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                rk1, rk2, score = line.split(',')
                self.id_list.append([rk1, rk2, score])

    def get_record(self, id):
        _, tfrecord_data_file, start_offset, end_offset = get_index_by_id(self.data_info, id)
        row = read_single_record_with_spec_index(tfrecord_data_file, start_offset, end_offset,
                                                 self.desc)
        _, record = prese_row(row, self.desc)

        return record

    def __getitem__(self, index):
        id1, id2, score = self.id_list[index]
        record1, record2 = self.get_record(int(id1)), self.get_record(int(id2))
        data = {}
        data['video_feature1'] = record1['frame_feature']
        data['title1'] = record1['title']
        data['mask1'] = record1['frame_mask']
        data['title_asr1'] = record1['title_asr']

        data['video_feature2'] = record2['frame_feature']
        data['title2'] = record2['title']
        data['mask2'] = record2['frame_mask']
        data['title_asr2'] = record2['title_asr']

        data['score'] = float(score)

        return data

    def __len__(self):
        return len(self.id_list)


class ProtoIndexDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, data_info, iteration):
        self.cfg = cfg
        self.data_info = data_info
        desc_path = cfg["DESC_PATH"]
        with open(desc_path) as fh:
            self.desc = json.load(fh)

        self.iteration = iteration
        self.num_per_class = cfg['PROTO']['SUPPORT_SET'] + cfg['PROTO']['QUERY_SET']
        self.dict = json.load(open(cfg['PROTO']['PROTO_DICT']))
        self.tags = list(self.dict.keys())
        length = []
        for tag in self.tags:
            length.append(self.dict[tag]['length'])
        length = np.array(length)
        self.proba = length / np.sum(length)

    def __getitem__(self, _idx):
        sample_class = np.random.choice(len(self.tags),
                                        self.cfg['PROTO']['SAMPLE_CLASS'],
                                        replace=False,
                                        p=self.proba)
        index = []
        label = []
        for i in range(self.cfg['PROTO']['SAMPLE_CLASS']):  # i就是当前的label
            index.extend(
                random.sample(self.dict[self.tags[sample_class[i]]]['index_list'],
                              self.num_per_class))
            label.extend(torch.full((self.num_per_class, ), i))
        label = torch.stack(label)

        frame_feature = []
        mask = []
        for i in index:
            _, tfrecord_data_file, start_offset, end_offset = get_index_by_index(self.data_info, i)
            row = read_single_record_with_spec_index(tfrecord_data_file, start_offset, end_offset,
                                                     self.desc)
            _, record = prese_row(row, self.desc)
            frame_feature.append(record['frame_feature'])
            mask.append(record['frame_mask'])
        frame_feature = np.stack(frame_feature)
        frame_feature = torch.tensor(frame_feature)
        mask = np.stack(mask)
        mask = torch.tensor(mask)

        return frame_feature, mask, label

    def __len__(self):
        return len(self.iteration)


class DoubletPairwiseDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, tsv_path, data_info):
        self.cfg = cfg
        self.id_list = []
        self.data_info = data_info
        desc_path = cfg["DESC_PATH"]
        with open(desc_path) as fh:
            self.desc = json.load(fh)

        with open(tsv_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                rk1, rk2, score = line.split(',')
                self.id_list.append([rk1, rk2, score])

    def get_record(self, id):
        _, tfrecord_data_file, start_offset, end_offset = get_index_by_id(self.data_info, id)
        row = read_single_record_with_spec_index(tfrecord_data_file, start_offset, end_offset,
                                                 self.desc)
        num_frames, record = prese_row(row, self.desc)

        return num_frames, record

    def __getitem__(self, index):
        # 重复采样，直到两个对的score不同，令第一个对的score大于第二个
        while True:
            doublet1, doublet2 = random.sample(self.id_list, 2)
            score12 = doublet1[2]
            score34 = doublet2[2]
            # if score12 == score34:
            #     continue
            # elif score12 < score34:
            #     doublet1, doublet2 = doublet2, doublet1
            #     break
            # else:
            #     break
            if score12 >= score34:
                break
            else:
                doublet1, doublet2 = doublet2, doublet1
                break

        id1, id2, score12 = doublet1
        id3, id4, score34 = doublet2
        ids = [id1, id2, id3, id4]

        data = {}
        for i in range(4):
            num_frames, record = self.get_record(int(ids[i]))
            video_feature = record['frame_feature']
            aug = np.random.randint(3)
            if aug == 0:
                aug_idx = [x for x in range(num_frames)]
                np.random.shuffle(aug_idx)
                video_feature[aug_idx]
            data['video_feature' + str(i + 1)] = video_feature

            data['title' + str(i + 1)] = record['title']
            data['title_asr' + str(i + 1)] = record['title_asr']
            data['mask' + str(i + 1)] = record['frame_mask']

        data['score12'] = float(score12)
        data['score34'] = float(score34)

        return data

    def __len__(self):
        return self.cfg['ITERATION']
