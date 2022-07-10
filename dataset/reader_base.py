#%%
import typing
import io
import os
import numpy as np
from numpy.lib.shape_base import tile
import pandas as pd
import struct
import warnings
import utils.example_pb2 as example_pb2
import concurrent.futures
from concurrent.futures import as_completed
# import jionlp as jio
from string import punctuation, digits
import re


# 删除无效以及重复字符
def del_invalid_character(s):
    rule = re.compile(u'[^a-zA-Z.,;《》？！“”‘’@#￥%…&×（）——+【】{};；●，。&～、|\s:：' + digits + punctuation +
                      '\u4e00-\u9fa5]+')
    s = re.sub(rule, '', s)
    s = re.sub('[、]+', '，', s)
    s = re.sub('\'', '', s)
    s = re.sub('[#]+', '，', s)
    s = re.sub('[?]+', '？', s)
    s = re.sub('[;]+', '，', s)
    s = re.sub('[,]+', '，', s)
    s = re.sub('[!]+', '！', s)
    s = re.sub('[.]+', '.', s)
    s = re.sub('[，]+', '，', s)
    s = re.sub('[。]+', '。', s)
    s = re.sub('[ ]+', ' ', s)
    # s = re.sub('[0-9]+$', ' ', s)
    s = re.sub('[\n\t\r]+', '', s)
    s = s.strip().lower()
    s = re.sub(r'([\u4e00-\u9fa5])\1{2,}', r'\1', s)  #去除重复中文
    s = re.sub(r'\W[\u4e00-\u9fa5]{1,2}(\W)', r'\1', s)  #去除无效字符
    s = re.sub(r'\W[\u4e00-\u9fa5]{1,2}(\W)', r'\1', s)
    if s == 'nan':
        s = ''
    return s


# 数据清洗
def clean_data(text: str):
    # 去除 html 标签、去除异常字符、去除冗余字符、去除括号补充内容、去除 URL、去除 E-mail、去除电话号码，将全角字母数字空格替换为半角
    # text = jio.clean_text(text)
    # 去除重复符号 以及换行
    text = del_invalid_character(text)
    return text


# 生成下标文件
def create_index(tfrecord_file: str, index_file: str) -> None:
    """Create index from the tfrecords file.

    Stores starting location (byte) and length (in bytes) of each
    serialized record.

    Params:
    -------
    tfrecord_file: str
        Path to the TFRecord file.

    index_file: str
        Path where to store the index file.
    """
    infile = open(tfrecord_file, "rb")
    outfile = open(index_file, "w")

    while True:
        current = infile.tell()
        try:
            byte_len = infile.read(8)
            if len(byte_len) == 0:
                break
            infile.read(4)
            proto_len = struct.unpack("q", byte_len)[0]
            infile.read(proto_len)
            infile.read(4)
            outfile.write(str(current) + " " + str(infile.tell() - current) + "\n")
        except:
            print("Failed to parse TFRecord.")
            break
    infile.close()

    outfile.close()


# 获取数据的下标
def get_index(data_file, index_file=""):
    if index_file == "":
        tmp_index_file = os.path.splitext(data_file)[0] + '.index'
        create_index(data_file, tmp_index_file)
        index_file = tmp_index_file
    index = np.loadtxt(index_file, dtype=np.int64)[:, 0]
    return index


# 获得数据长度
def get_data_len(data_index_list):
    data_len = 0
    for data_index in data_index_list:
        data_len += len(data_index)
    return data_len


# 通过下标获得行所在文件id 以及文件起止位置
def get_nth_data_file(index, data_index_list):
    '''
    :param index: index of target item
    :return: item file index, and item start & end offset
    '''
    for i, data_index in enumerate(data_index_list):
        if index < len(data_index):
            break
        index -= len(data_index)
    start_offset = data_index[index]
    end_offset = data_index[index + 1] if index + 1 < len(data_index) else None
    return i, (start_offset, end_offset)


# 通过下标读一行数据
def read_single_record_with_spec_index(
    data_path: str,
    start_offset: int,
    end_offset: int,
    description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
) -> typing.Dict:
    """
    Read data from tfrecord dataset with start_offset and end_offset.

    Params:
    -------
    data_path: str
        TFRecord file path.

    start_offset: int
        start offset. Can be set to None if read from scratch.

    end_offset: int
        end offset. Can be set to None if read to the end.

    description: list or dict of str, optional, default=None
        List of keys or dict of (key, value) pairs to extract from each
        record. The keys represent the name of the features and the
        values ("byte", "float", or "int") correspond to the data type.
        If dtypes are provided, then they are verified against the
        inferred type for compatibility purposes. If None (default),
        then all features contained in the file are extracted.

    :return:
    -------
    datum_bytes_view: memoryview
        Object referencing the specified `datum_bytes` contained in the
        file (for a single record).
    """
    typename_mapping = {
        "byte": "bytes_list",
        "bytes": "bytes_list",
        "float": "float_list",
        "int": "int64_list"
    }

    file = io.open(data_path, "rb")

    record = None

    if start_offset is not None:
        file.seek(start_offset)
    if end_offset is None:
        end_offset = os.path.getsize(data_path)
    if file.tell() < end_offset:
        byte_len = file.read(8)
        if len(byte_len) <= 0:
            raise RuntimeError("Invalid byte_len.")
        file.read(4)
        proto_len = struct.unpack("q", byte_len)[0]
        record = file.read(proto_len)
        if len(record) != proto_len:
            raise RuntimeError("Failed to read the record.")

    file.close()

    if record is None:
        raise RuntimeError("Seek with wrong start_offset.")

    return read_single_description(description, record, typename_mapping)


# 读一行数据
def read_single_description(description, record, typename_mapping):
    example = example_pb2.Example()
    example.ParseFromString(record)

    all_keys = list(example.features.feature.keys())
    if description is None:
        description = dict.fromkeys(all_keys, None)
    elif isinstance(description, list):
        description = dict.fromkeys(description, None)

    features = {}
    for key, typename in description.items():
        if key not in all_keys:
            warnings.warn(f"Key {key} doesn't exist (select from {all_keys})!", RuntimeWarning)
            continue
        # NOTE: We assume that each key in the example has only one field
        # (either "bytes_list", "float_list", or "int64_list")!
        field = example.features.feature[key].ListFields()[0]
        inferred_typename, value = field[0].name, field[1].value
        if typename is not None:
            tf_typename = typename_mapping[typename]
            if tf_typename != inferred_typename:
                reversed_mapping = {v: k for k, v in typename_mapping.items()}
                raise TypeError(f"Incompatible type '{typename}' for `{key}` "
                                f"(should be '{reversed_mapping[inferred_typename]}').")

        # Decode raw bytes into respective data types
        if typename == "byte":
            value = np.frombuffer(value[0], dtype=np.uint8)
        elif typename == "bytes":
            value = [np.frombuffer(v, dtype=np.uint8) for v in value]
        elif typename == "float":
            value = np.array(value, dtype=np.float32)
        elif typename == "int":
            value = np.array(value, dtype=np.int32)
        features[key] = value

    return features


# 解析一行数据
def prese_row(row, desc, training=False, clean=False):
    record = {}
    for key, value in row.items():
        if key == "frame_feature":
            num_segments = 32  #32帧数
            frame_feature = [
                np.frombuffer(bytes(x), dtype=np.float16).astype(np.float32) for x in value
            ]
            zero_frame = frame_feature[0] * 0.
            num_frames = len(frame_feature)
            frame_gap = (num_frames - 1) / num_segments
            if frame_gap <= 1:
                res = frame_feature + [zero_frame] * (num_segments - num_frames)
            else:
                # 有什么作用？
                if training:
                    res = [
                        frame_feature[round(i * frame_gap + np.random.uniform(0, frame_gap))]
                        for i in range(num_segments)
                    ]
                else:
                    res = [frame_feature[round((i + 0.5) * frame_gap)] for i in range(num_segments)]
            value = np.c_[res]
            record[key] = value
        else:
            type_name = desc[key]
            if type_name == "byte":
                value = bytes(value).decode()
            elif type_name == "bytes":
                value = [bytes(v).decode() for v in value]
            elif type_name == "float":
                value = value
            elif type_name == "int":
                value = value
            record[key] = value
    mask = np.zeros(32, dtype=np.int32)
    mask[:num_frames] = 1
    record['frame_mask'] = (mask == 1)
    if clean:
        record['title'] = clean_data(record['title'])
        record['asr_text'] = clean_data(record['asr_text'])
        if len(record['asr_text']) < 10:
            record['asr_text'] = ''
    record['title_asr'] = [record['title'], record['asr_text']]
    return num_frames, record


def get_index_list(file_path_list):
    # 线程数
    max_workers = min(8, len(file_path_list))

    # 获得所有行的起始位置
    data_index_list = [None] * len(file_path_list)
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        fs = {
            executor.submit(get_index, data_path): i
            for i, data_path in enumerate(file_path_list)
        }
        for future in as_completed(fs):
            data_index_list[fs[future]] = future.result()
    return data_index_list


def generate_info(file_path_list, desc, save_path="result/data_info.csv"):
    data_index_list = get_index_list(file_path_list)
    data_len = get_data_len(data_index_list)

    result = []
    for i in range(data_len):
        data_file_index, (start_offset, end_offset) = get_nth_data_file(i, data_index_list)
        tfrecord_data_file = file_path_list[data_file_index]
        row = read_single_record_with_spec_index(tfrecord_data_file, start_offset, end_offset, desc)
        frame_num, record = prese_row(row, desc)
        print(i, record['id'])
        # result.append((record['id'], file_path_list[data_file_index], start_offset, end_offset))
        result.append((record['id'], frame_num, record['title'], record['asr_text'],
                       record['category_id'], record['tag_id']))

    # %%
    # df = pd.DataFrame(result, columns=["id", "file_path", "start_offset", "end_offset"])
    df = pd.DataFrame(result,
                      columns=["id", "frame_num", "title", "asr_text", "category_id", "tag_id"])
    df.to_csv(save_path, index=0)


def get_index_by_id(data_info, id):
    _, tfrecord_data_file, start_offset, end_offset = data_info.loc[data_info["id"] == id].iloc[0]
    end_offset = None if np.isnan(end_offset) else end_offset
    return _, tfrecord_data_file, start_offset, end_offset


def get_index_by_index(data_info, index):
    _, tfrecord_data_file, start_offset, end_offset = data_info.iloc[index][[
        "id", "file_path", "start_offset", "end_offset"
    ]]
    end_offset = None if np.isnan(end_offset) else end_offset
    return _, tfrecord_data_file, start_offset, end_offset