import time
import logging
from tempfile import TemporaryDirectory

import onnxruntime
import numpy as np
from scipy.special import softmax
from transformers import AutoTokenizer


logger = logging.getLogger(__name__)


def print_run_time(func):  
    def wrapper(*args, **kw):  
        local_time = time.time()  
        res = func(*args, **kw)
        logger.info ('cost {0:.4f}s to run {1}'.format(time.time() - local_time, func.__name__))
        return res
    return wrapper


class TokenClassifer(object):
    def __init__(self, tokenizer_dir, model_path, model, label_list) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir, cache_dir=TemporaryDirectory()
        )
        self.session = onnxruntime.InferenceSession(
            model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        logger.info("using providers: {}".format(self.session.get_providers()))
        if model in ["distilbert"]:
            self.input_ids_name, self.attention_mask_name = [ele.name for ele in self.session.get_inputs()]
        elif model in ["bert"]:
            self.input_ids_name, self.attention_mask_name, self.token_type_ids_name = [ele.name for ele in self.session.get_inputs()]
        else:
            raise Exception("model-{} not implemented".format(model))
        self.model = model
        self.label_list = label_list

    @print_run_time
    def predict(self, data, **kwargs):
        """
        通过tokenizer处理输入的字符串，并通过onnx对token进行分类

        Parameters
        ----------
        data : dict
            输入的数据
        data.tokens : list[str]
            输入的句子列表

        Returns
        -------
        result : list[dict]
            token的分类结果
        result[*].raw_str : str
            原始字符串
        result[*].ne : list[dict]
            对token的分类
        result[*].ne[*].tag : str
            token的预测类别
        result[*].ne[*].offset : int
            token在原始字符串中的起始位置index
        result[*].ne[*].length : int
            token的字符串长度
        result[*].ne[*].text : str
            token字符串
        result[*].ne[*].score : float
            token的预测分数，如果这个字符串由多个token组成，则取scores的最小值
        """
        tokenized_inputs = self.tokenizer(data["tokens"], padding=True, truncation=True, is_split_into_words=False)
        input_ids = np.array(tokenized_inputs["input_ids"], dtype=np.int64)
        attention_mask = np.array(tokenized_inputs["attention_mask"], dtype=np.int64)
        if self.model in ["distilbert"]:
            predictions = self.session.run([], 
                {
                    self.input_ids_name: input_ids,
                    self.attention_mask_name: attention_mask
                }
            )[0]
        elif self.model in ["bert"]:
            token_type_ids = np.array(tokenized_inputs["token_type_ids"], dtype=np.int64)
            predictions = self.session.run([], 
                {
                    self.input_ids_name: input_ids,
                    self.attention_mask_name: attention_mask,
                    self.token_type_ids_name: token_type_ids
                }
            )[0]
        scores = softmax(predictions, axis=-1)
        max_indexes = np.argmax(scores, axis=-1)
        results = []
        for j in range(len(data["tokens"])):
            pred = self.parse_result(tokenized_inputs, j, data["tokens"][j], scores[j], max_indexes[j])
            results.append({
                "raw_str": data["tokens"][j], "ne": pred
            })
        return results
        
    def parse_result(self, tokenized_inputs, batch_index, raw_str, scores, max_indexes):
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
        max_indexes = [max_indexes[i] for i in keep_indexes]
        scores = [scores[i] for i in keep_indexes]
        scores = [scores[i][j] for i, j in enumerate(max_indexes)]

        pred_ne = []
        for i, word_id in enumerate(word_ids):
            if self.label_list[max_indexes[i]] == 'O':
                continue
            if encode_tokens[i].startswith("##") or self.label_list[max_indexes[i]].startswith("I-"):
                if len(pred_ne) > 0:
                    pred_ne[-1]["length"] = offsets[i][1] - pred_ne[-1]["offset"]
                    pred_ne[-1]["text"] = raw_str[pred_ne[-1]["offset"]:pred_ne[-1]["offset"] + pred_ne[-1]["length"]]
                    pred_ne[-1]["score"] = min(pred_ne[-1]["score"], scores[i])
                else:
                    pred_ne.append({
                        "tag": self.label_list[max_indexes[i]][2:],
                        "offset": offsets[i][0],
                        "length": offsets[i][1] - offsets[i][0],
                        "text": raw_str[offsets[i][0]:offsets[i][1]],
                        "score": scores[i]
                    })
            elif self.label_list[max_indexes[i]].startswith("B-"):
                pred_ne.append({
                    "tag": self.label_list[max_indexes[i]][2:],
                    "offset": offsets[i][0],
                    "length": offsets[i][1] - offsets[i][0],
                    "text": raw_str[offsets[i][0]:offsets[i][1]],
                    "score": scores[i]
                })
        
        return pred_ne
        