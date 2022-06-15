import os
import glob
import json
import math
import logging

import numpy as np
from scipy.special import softmax
import onnxruntime
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
# from datasets import set_caching_enabled
# set_caching_enabled(False) # disable并不是不产生cache，而是不使用cache
from transformers import AutoTokenizer

from .utils import convert_from_euler_to_huggingface, read_job_config, tokenize_and_align_labels


logger = logging.getLogger(__name__)


def get_configs():
    """
    获取配置
    """
    job_configs = read_job_config()
    logger.info("job_configs:{}".format(job_configs))
    # 读取input：
    for parameter in job_configs['input']:
        if parameter['name'] == 'test_file':
            # eg. "/data/conll2003/test_euler.json"
            test_file = parameter['value']

        if parameter['name'] == 'label':
            # eg. "/data/conll2003/label_euler.txt"
            label_file = parameter['value']
        
        if parameter['name'] == 'model_ckpt_dir':
            # eg. "/data/conll2003/debug_train/checkpoints"
            model_ckpt = parameter['value']

        if parameter['name'] == 'onnx_path':
            # eg. "/data/conll2003/debug_train/onnx/best.onnx"
            onnx_path = parameter['value']

    # 读取output：
    for parameter in job_configs['output']:
        if parameter['name'] == 'result_path':
            # eg. "/data/conll2003/debug_train/result.json"
            result_path = parameter['value']
            if not os.path.exists(os.path.dirname(result_path)):
                os.makedirs(os.path.dirname(result_path))

    kwargs = job_configs.get('args', {})

    configs = {
        "test_file": test_file,
        "label_list": label_file,
        "model_ckpt": model_ckpt,
        "onnx_path": onnx_path,
        "batch_size": kwargs["batch_size"],
        "label_col": "tags",
        "result_path": result_path,
        "from_euler": kwargs.get("from_euler", True),
        "model": kwargs["model"]
    }
    return configs


def data_generator(test_data, configs, batch_size=1):
    cnt = 0
    token_batch = [None]*batch_size
    label_batch = [None]*batch_size
    text_batch = [None]*batch_size
    offset_batch = [None]*batch_size
    length_batch = [None]*batch_size
    while cnt < len(test_data):
        token_batch[cnt%batch_size] = test_data[cnt]["tokens"]
        label_batch[cnt%batch_size] = test_data[cnt][configs["label_col"]]
        text_batch[cnt%batch_size] = test_data[cnt]["raw_str"]
        offset_batch[cnt%batch_size] = test_data[cnt]["offsets"]
        length_batch[cnt%batch_size] = test_data[cnt]["lengths"]
        cnt += 1
        if cnt % batch_size == 0:
            yield {
                "tokens": token_batch, configs["label_col"]: label_batch, "raw_str": text_batch,
                "offsets": offset_batch, "lengths": length_batch
            }
            token_batch = [None]*batch_size
            label_batch = [None]*batch_size
            text_batch = [None]*batch_size
            offset_batch = [None]*batch_size
            length_batch = [None]*batch_size
    
    if cnt % batch_size != 0:
        token_batch = token_batch[:cnt%batch_size]
        label_batch = label_batch[:cnt%batch_size]
        text_batch = text_batch[cnt%batch_size]
        offset_batch = offset_batch[cnt%batch_size]
        length_batch = length_batch[cnt%batch_size]
        yield {
                "tokens": token_batch, configs["label_col"]: label_batch, "raw_str": text_batch,
                "offsets": offset_batch, "lengths": length_batch
            }


def pad_input_data(tokenized_inputs):
    max_len = max(*[len(ele) for ele in tokenized_inputs["input_ids"]])
    for i in range(len(tokenized_inputs["input_ids"])):
        tokenized_inputs["input_ids"][i] = tokenized_inputs["input_ids"][i] + [0]*(max_len - len(tokenized_inputs["input_ids"][i]))
    
    max_len = max(*[len(ele) for ele in tokenized_inputs["attention_mask"]])
    for i in range(len(tokenized_inputs["attention_mask"])):
        tokenized_inputs["attention_mask"][i] = tokenized_inputs["attention_mask"][i] + [0]*(max_len - len(tokenized_inputs["attention_mask"][i]))
    
    if "token_type_ids" in tokenized_inputs:
        max_len = max(*[len(ele) for ele in tokenized_inputs["token_type_ids"]])
        for i in range(len(tokenized_inputs["token_type_ids"])):
            tokenized_inputs["token_type_ids"][i] = tokenized_inputs["token_type_ids"][i] + [0]*(max_len - len(tokenized_inputs["token_type_ids"][i]))

    max_len = max(*[len(ele) for ele in tokenized_inputs["labels"]])
    for i in range(len(tokenized_inputs["labels"])):
        tokenized_inputs["labels"][i] = tokenized_inputs["labels"][i] + [-100]*(max_len - len(tokenized_inputs["labels"][i]))
    return tokenized_inputs


def compute_metrics(true_predictions, true_labels):
    """
    计算模型指标
    """
    results = {}
    for metric in [accuracy_score, precision_score, recall_score, f1_score]:
        results[metric.__name__] = metric(true_labels, true_predictions)
    return results


def parse_result(tokenized_inputs, batch_index, batch, prediction, label):
    """
    解析结果到euler的格式
    """
    word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
    keep_indexes = [i for i, word_id in enumerate(word_ids) if word_id is not None]
    word_ids = [word_ids[i] for i in keep_indexes]
    offsets = tokenized_inputs.encodings[batch_index].offsets
    offsets = [offsets[i] for i in keep_indexes]
    encode_tokens = tokenized_inputs.encodings[batch_index].tokens
    encode_tokens = [encode_tokens[i] for i in keep_indexes]
    word_offsets = batch["offsets"][batch_index]
    raw_str = batch["raw_str"][batch_index]

    pred_ne = []
    gt_ne = []
    for i, word_id in enumerate(word_ids):
        if prediction[i] == 'O' and label[i] == 'O':
            continue
        if encode_tokens[i].startswith("##") or prediction[i].startswith("I-"):
            if len(pred_ne) > 0:
                pred_ne[-1]["length"] = word_offsets[word_id] + offsets[i][1] - pred_ne[-1]["offset"]
                pred_ne[-1]["text"] = raw_str[pred_ne[-1]["offset"]:pred_ne[-1]["offset"] + pred_ne[-1]["length"]]
            else:
                pred_ne.append({
                    "tag": prediction[i][2:],
                    "offset": word_offsets[word_id] + offsets[i][0],
                    "length": offsets[i][1] - offsets[i][0],
                    "text": raw_str[word_offsets[word_id] + offsets[i][0]:word_offsets[word_id] + offsets[i][1]]
                })
        elif prediction[i].startswith("B-"):
            pred_ne.append({
                "tag": prediction[i][2:],
                "offset": word_offsets[word_id] + offsets[i][0],
                "length": offsets[i][1] - offsets[i][0],
                "text": raw_str[word_offsets[word_id] + offsets[i][0]:word_offsets[word_id] + offsets[i][1]]
            })

        if encode_tokens[i].startswith("##") or label[i].startswith("I-"):
            gt_ne[-1]["length"] = word_offsets[word_id] + offsets[i][1] - gt_ne[-1]["offset"]
            gt_ne[-1]["text"] = raw_str[gt_ne[-1]["offset"]:gt_ne[-1]["offset"] + gt_ne[-1]["length"]]
        elif label[i].startswith("B-"):
            gt_ne.append({
                "tag": label[i][2:],
                "offset": word_offsets[word_id] + offsets[i][0],
                "length": offsets[i][1] - offsets[i][0],
                "text": raw_str[word_offsets[word_id] + offsets[i][0]:word_offsets[word_id] + offsets[i][1]]
            })
    
    return pred_ne, gt_ne


def main(configs):
    logger.info("load pretrained AutoTokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        configs["model_ckpt"], cache_dir=configs["cache_dir"]
    )
    batch_size = configs["batch_size"]
    session = onnxruntime.InferenceSession(configs["onnx_path"])
    if configs["model"] in ["distilbert"]:
        input_ids_name, attention_mask_name = [ele.name for ele in session.get_inputs()]
    elif configs["model"] in ["bert"]:
        input_ids_name, attention_mask_name, token_type_ids_name = [ele.name for ele in session.get_inputs()]
    else:
        raise Exception("model-{} not implemented".format(configs["model"]))

    if configs["from_euler"]:
        configs["test_file"], configs["label_list"] = convert_from_euler_to_huggingface(
            configs["test_file"], configs["label_list"], tag_name=configs["label_col"]
        )
    with open(configs["test_file"], "r") as f:
        test_data = json.load(f)["data"]
    with open(configs["label_list"], "r") as f:
        label_list = [ele.strip() for ele in f.readlines()]
    total_batch = math.ceil(len(test_data) / batch_size)

    true_predictions = []
    true_labels = []
    for i, batch in enumerate(data_generator(test_data, configs, batch_size=batch_size)):
        if (i+1) % 10 == 0:
            logger.info("{}/{} batches processed".format(i+1, total_batch))
        tokenized_inputs = tokenize_and_align_labels(batch, tokenizer, label_col=configs["label_col"])
        tokenized_inputs = pad_input_data(tokenized_inputs)
        input_ids = np.array(tokenized_inputs["input_ids"], dtype=np.int64)
        attention_mask = np.array(tokenized_inputs["attention_mask"], dtype=np.int64)
        if configs["model"] in ["distilbert"]:
            predictions = session.run([], 
                {
                    input_ids_name: input_ids,
                    attention_mask_name: attention_mask
                }
            )[0]
        elif configs["model"] in ["bert"]:
            token_type_ids = np.array(tokenized_inputs["token_type_ids"], dtype=np.int64)
            predictions = session.run([], 
                {
                    input_ids_name: input_ids,
                    attention_mask_name: attention_mask,
                    token_type_ids_name: token_type_ids
                }
            )[0]
        # predictions = softmax(predictions, axis=-1)
        predictions = np.argmax(predictions, axis=-1)
        labels = tokenized_inputs["labels"]
        true_predictions += [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels += [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        if i == 0:
            batch_results = []
            for j in range(len(batch["tokens"])):
                pred, gt = parse_result(tokenized_inputs, j, batch, true_predictions[j], true_labels[j])
                batch_results.append({
                    "raw_str": batch["raw_str"][j], "prediction": pred, "gt": gt
                })
            logger.info("show batch-0 results: {}".format(batch_results))
            
    result = compute_metrics(true_predictions, true_labels)
    logger.info("result:{}".format(result))
    with open(configs["result_path"], "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
        datefmt='%Y-%m-%d %A %H:%M:%S'
    )
    configs = get_configs()
    configs["cache_dir"] = os.path.dirname(configs["model_ckpt"])
    checkpoints = glob.glob(os.path.join(configs["model_ckpt"], "*"))
    checkpoints = [checkpoint for checkpoint in checkpoints if os.path.basename(checkpoint).startswith('checkpoint')]
    if len(checkpoints) == 1:
        configs["model_ckpt"] = checkpoints[0]
    elif len(checkpoints) > 1:
        checkpoints = sorted(checkpoints, key=lambda x:int(x.split('-')[-1]))
        configs["model_ckpt"] = checkpoints[-1]
    else:
        pass
    main(configs)
