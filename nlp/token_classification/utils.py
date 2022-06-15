import os
import re
import json
import logging


logger = logging.getLogger(__name__)


def get_strip_token(token):
    strip_token = token.strip()
    if len(strip_token) == 0:
        return "", 0
    offset = 0
    for i in range(len(token)):
        if token[i] != strip_token[0]:
            offset += 1
        else:
            break
    return strip_token, offset


def is_english_char(s):
    if re.match(r'[a-zA-Z]+', s):
        return True
    else:
        return False


def is_number(s):
    """
    是否是数字
    """
    if re.match(r'\d+', s):
        return True
    else:
        return False


def convert_from_euler_to_huggingface(data_file, label_file, tag_name="tags", change_label_file=True):
    """
    数据格式转换，从euler格式转为huggingface

    Parameters
    ----------
    data_file : str
        euler格式的数据所在的json文件路径
    label_file : str
        标签文件路径

    Returns
    -------
    result : str
        结果文件路径, 存在和data_file同样的目录下
    """
    with open(data_file, "r") as f:
        raw_data = json.load(f)
    with open(label_file, "r") as f:
        label_list = [ele.strip() for ele in f.readlines()]
    basename = os.path.basename(data_file)
    basename = ".".join(basename.split(".")[:-1]) + "_hug.json"
    save_file_path = os.path.join(os.path.dirname(data_file), basename)
    if change_label_file:
        basename = os.path.basename(label_file)
        basename = ".".join(basename.split(".")[:-1]) + "_hug.txt"
        save_label_path = os.path.join(os.path.dirname(label_file), basename)
        with open(save_label_path, "w") as f:
            # euler没有"B-"和"I-"，huggingface要添加
            for i in range(len(label_list)):
                if label_list[i] == 'O':
                    f.write(label_list[i])
                else:
                    f.write("B-" + label_list[i] + '\n')
                    f.write("I-" + label_list[i])
                f.write('\n')
        logger.info("save converted label to {}".format(save_label_path))
        with open(save_label_path, "r") as f:
            label_list = [ele.strip() for ele in f.readlines()]
    else:
        save_label_path = label_file

    result = {"data": []}
    for idx, item in enumerate(raw_data):
        ne = sorted(item["ne"], key=lambda x:x["offset"])
        start = 0
        offsets = []
        lengths = []
        tokens = []
        tags = []
        for ele in ne:
            if ele["offset"] == start:
                token = item["raw_str"][start:start+ele["length"]]
                strip_token, offset = get_strip_token(token)
                if len(strip_token) > 0:
                    if ele["tag"] == "O":
                        tokens.append(strip_token)
                        offsets.append(start+offset)
                        lengths.append(len(token))
                        tags.append(label_list.index("O"))
                    else:
                        if is_english_char(strip_token[0]):
                            split = len(strip_token.split(' ')[0])
                        elif is_number(strip_token[0]):
                            split = 1
                            j = 1
                            while j < len(strip_token) and is_number(strip_token[j]):
                                split += 1
                                j += 1
                        else:
                            split = 1

                        # 创建B-
                        tokens.append(strip_token[:split])
                        offsets.append(start+offset)
                        lengths.append(split)
                        tags.append(label_list.index("B-"+ele["tag"]))
                        
                        # 创建I-
                        if len(strip_token[split:]) > 0:
                            strip_token2, offset2 = get_strip_token(strip_token[split:])
                            tokens.append(strip_token2)
                            offsets.append(start+offset+split+offset2)
                            lengths.append(len(strip_token2))
                            tags.append(label_list.index("I-"+ele["tag"]))
                start = start + ele["length"]
            elif ele["offset"] > start:
                token = item["raw_str"][start:ele["offset"]]
                strip_token, offset = get_strip_token(token)
                if len(strip_token) > 0:
                    tokens.append(strip_token)
                    offsets.append(start+offset)
                    lengths.append(len(token))
                    tags.append(label_list.index("O"))
                
                start = ele["offset"]
                token = item["raw_str"][start:start+ele["length"]]
                strip_token, offset = get_strip_token(token)
                if len(strip_token) > 0:
                    if ele["tag"] == "O":
                        tokens.append(strip_token)
                        offsets.append(start+offset)
                        lengths.append(len(token))
                        tags.append(label_list.index("O"))
                    else:
                        if is_english_char(strip_token[0]):
                            split = len(strip_token.split(' ')[0])
                        elif is_number(strip_token[0]):
                            split = 1
                            j = 1
                            while j < len(strip_token) and is_number(strip_token[j]):
                                split += 1
                                j += 1
                        else:
                            split = 1

                        # 创建B-
                        tokens.append(strip_token[:split])
                        offsets.append(start+offset)
                        lengths.append(split)
                        tags.append(label_list.index("B-"+ele["tag"]))
                        
                        # 创建I-
                        if len(strip_token[split:]) > 0:
                            strip_token2, offset2 = get_strip_token(strip_token[split:])
                            tokens.append(strip_token2)
                            offsets.append(start+offset+split+offset2)
                            lengths.append(len(strip_token2))
                            tags.append(label_list.index("I-"+ele["tag"]))
                start = start + ele["length"]
        if start < len(item["raw_str"]):
            token = item["raw_str"][start:]
            strip_token, offset = get_strip_token(token)
            if len(strip_token) > 0:
                tokens.append(strip_token)
                offsets.append(start+offset)
                lengths.append(len(token))
                tags.append(label_list.index("O"))
        result["data"].append(
            {"id": idx, "tokens": tokens, tag_name: tags, "offsets": offsets, "lengths": lengths, "raw_str": item["raw_str"]}
        )
    with open(save_file_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("save converted data to {}".format(save_file_path))
    return save_file_path, save_label_path


def restore_string(ne, tokens):
    """
    还原原始字符串

    Parameters
    ----------
    ne : list[dict]
        每一个token在原始字符串中的offset和length
    tokens : list[str]
        句子中所有的tokens

    Returns
    -------
    raw_str : str
        原始字符串
    """
    if len(ne) == 0:
        raw_str = ""
    else:
        raw_str = " " * (ne[-1]["offset"] + ne[-1]["length"])
        for i, n in enumerate(ne):
            raw_str = raw_str[:n["offset"]] + tokens[i] + raw_str[n["offset"]+len(tokens[i]):]
    return raw_str


def convert_from_huggingface_to_euler(data_file, label_file, tag_name="ner_tags", change_label_file=True):
    """
    数据格式转换，从huggingface格式转为euler

    Parameters
    ----------
    data_file : str
        huggingface格式的数据所在的json文件路径
    label_file : str
        标签文件路径

    Returns
    -------
    result : str
        结果文件路径, 存在和data_file同样的目录下
    """
    with open(data_file, "r") as f:
        raw_data = json.load(f)
    with open(label_file, "r") as f:
        label_list = [ele.strip() for ele in f.readlines()]
    
    basename = os.path.basename(data_file)
    basename = ".".join(basename.split(".")[:-1]) + "_euler.json"
    save_file_path = os.path.join(os.path.dirname(data_file), basename)
    if change_label_file:
        basename = os.path.basename(label_file)
        basename = ".".join(basename.split(".")[:-1]) + "_euler.txt"
        save_label_path = os.path.join(os.path.dirname(label_file), basename)
        with open(save_label_path, "w") as f:
            # huggingface的标签有"B-"和"I-"，euler的标签需要合并这两类
            for i in range(len(label_list)):
                if label_list[i] == 'O':
                    f.write(label_list[i])
                elif label_list[i].startswith("B-"):
                    f.write(label_list[i][2:])
                else:
                    continue
                f.write('\n')
        logger.info("save converted label to {}".format(save_label_path))
    else:
        save_label_path = label_file

    result = []
    for item in raw_data["data"]:
        ne = []
        start = -1
        for idx in range(len(item[tag_name])):
            if "offsets" in item and "lengths" in item:
                ne.append({
                    "tag": label_list[item[tag_name][idx]],
                    "offset": item["offsets"][idx],
                    "length": item["lengths"][idx]
                })
            else:
                ne.append({
                    "tag": label_list[item[tag_name][idx]],
                    "offset": start + 1,
                    "length": len(item["tokens"][idx])
                })
            start = ne[-1]["offset"] + ne[-1]["length"]

        # 还原原始字符串
        raw_str = restore_string(ne, item["tokens"])
        
        # 合并B-和I-
        start = 0
        new_ne = []
        for n in ne:
            if n["tag"] == "O":
                new_ne.append(n)
            elif n["tag"].startswith("B-"):
                start = n["offset"]
                n["tag"] = n["tag"][2:]
                new_ne.append(n)
            else:
                new_ne[-1]["length"] = n["offset"] + n["length"] - start
        # 添加text
        ne = new_ne
        for n in ne:
            n["text"] = raw_str[n["offset"]:n["offset"]+n["length"]]

        result.append(
            {
                "raw_str": raw_str,
                "ne": ne
            }
        )
    with open(save_file_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("save converted data to {}".format(save_file_path))
    return save_file_path, save_label_path


def read_job_config() -> dict:
    """
    读取ENV：JOB_CONFIGS
    举个例子:
    export JOB_CONFIGS='{"input": [{"name": "test_input", "type":"file", "value":"/data/pony_demo/demo_input"}], "output": [{"name": "train", "type":"file", "value":"/data/pony_demo/demo_train"},{"name": "test", "type":"file", "value":"/data/pony_demo/demo_test"}], "args":{"ratio":0.7, "seed":2}}'

    """
    jc = os.getenv('JOB_CONFIGS')
    return json.loads(jc)


def tokenize_and_align_labels(examples, tokenizer, label_col, label_all_tokens=True):
    """
    对齐分词之后的labels
    """
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples["{}".format(label_col)]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                # 默认label的奇数位为B-，偶数位为O或者I-
                word_label = label[word_idx] + 1 if label[word_idx] % 2 == 1 else label[word_idx]
                label_ids.append(word_label if label_all_tokens else -100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


if __name__ == "__main__":
    # file_dir = os.path.dirname(__file__)
    # data_file = os.path.join(file_dir, "../../debugs/hug.json")
    # label_file = os.path.join(file_dir, "../../debugs/label.txt")
    # convert_from_huggingface_to_euler(data_file, label_file, tag_name="ner_tags")

    # data_file = os.path.join(file_dir, "../../debugs/euler.json")
    # label_file = os.path.join(file_dir, "../../debugs/label_euler.txt")
    # convert_from_euler_to_huggingface(data_file, label_file, tag_name="ner_tags")

    data_files = [
        "/data/conll2003/train.json",
        "/data/conll2003/val.json",
        "/data/conll2003/test.json"
    ]
    label_file = "/data/conll2003/label.txt"
    for data_file in data_files:
        save_file_path, save_label_path = convert_from_huggingface_to_euler(data_file, label_file, tag_name="ner_tags")
        print ("save_file_path:", save_file_path)
        print ("save_label_path:", save_label_path)
        save_file_path, save_label_path = convert_from_euler_to_huggingface(save_file_path, save_label_path, tag_name="tags")
        print ("save_file_path:", save_file_path)
        print ("save_label_path:", save_label_path)
