from google.protobuf.descriptor import EnumValueDescriptor
import torch
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, LabelEncoder
from dataset.reader_base import read_single_record_with_spec_index, prese_row, get_index_by_id, get_index_by_index


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
        self.ohc = LabelEncoder()
        self.ohc.fit(self.category_id.reshape(-1, 1))
        # category_id = pd.read_csv(cfg['CATEGORY_ID_COUNT_INFO'])
        # category_count = category_id.sort_values(by='count', ascending=False)
        # self.category_id = np.array(category_count.iloc[:cfg['CATEGORY_SIZE']]['category_id'])
        # self.ohc = OneHotEncoder()
        # self.ohc.fit(self.category_id.reshape(-1, 1))

    def transform(self, record):
        # tokenizer title

        # 筛选的tag
        tags = [t for t in record['tag_id'] if t in self.selected_tags]
        multi_hot = self.mlb.transform([tags])[0]
        record['tag_id'] = multi_hot

        # 编码category_id
        category_id = [c for c in [record['category_id']] if c in self.category_id]
        one_hot = self.ohc.transform(category_id)[0]
        record['category_id'] = one_hot
        # record['category_id'] = record['category_id'][0]
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


class TestIndexDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, index_list, data_info):
        self.cfg = cfg
        self.index_list = index_list
        self.data_info = data_info
        desc_path = cfg["TEST_DESC_PATH"]
        with open(desc_path) as fh:
            self.desc = json.load(fh)

    def __getitem__(self, index):
        index = self.index_list[index]
        _, tfrecord_data_file, start_offset, end_offset = get_index_by_index(self.data_info, index)
        row = read_single_record_with_spec_index(tfrecord_data_file, start_offset, end_offset,
                                                 self.desc)
        _, record = prese_row(row, self.desc)
        return record

    def __len__(self):
        return len(self.index_list)


class IdDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, id_list, data_info, test=False):
        self.cfg = cfg
        self.id_list = id_list
        self.data_info = data_info
        self.test = test
        desc_path = cfg["TEST_DESC_PATH"]
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


class MyDfPairwiseDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, data_path, data_info):
        self.cfg = cfg
        self.data_info = data_info
        desc_path = cfg["DESC_PATH"]
        with open(desc_path) as fh:
            self.desc = json.load(fh)
        self.df = pd.read_csv(data_path)

    def get_record(self, id):
        _, tfrecord_data_file, start_offset, end_offset = get_index_by_id(self.data_info, id)
        row = read_single_record_with_spec_index(tfrecord_data_file, start_offset, end_offset,
                                                 self.desc)
        _, record = prese_row(row, self.desc)

        return record

    def __getitem__(self, index):
        r = self.df.iloc[index][["id_1", "id_2", "score"]]
        id1, id2, score = r
        record1, record2 = self.get_record(id1), self.get_record(id2)
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
        return len(self.df.index)


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


class DfPairwiseDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, index_list, k_fold_info, data_info):
        self.cfg = cfg
        self.index_list = index_list
        self.k_fold_info = k_fold_info
        self.data_info = data_info
        desc_path = cfg["DESC_PATH"]
        with open(desc_path) as fh:
            self.desc = json.load(fh)

    def get_record(self, id):
        _, tfrecord_data_file, start_offset, end_offset = get_index_by_id(self.data_info, id)
        row = read_single_record_with_spec_index(tfrecord_data_file, start_offset, end_offset,
                                                 self.desc)
        _, record = prese_row(row, self.desc)

        return record

    def __getitem__(self, index):
        index = self.index_list[index]
        id1, id2, score = self.k_fold_info.iloc[index][["id_1", "id_2", "score"]]
        record1, record2 = self.get_record(id1), self.get_record(id2)
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
        return len(self.index_list)