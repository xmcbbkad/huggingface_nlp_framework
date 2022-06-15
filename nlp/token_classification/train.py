import os
import json
import time
import logging
from functools import partial
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import transformers
from datasets import load_dataset
from seqeval.metrics import accuracy_score, precision_score, recall_score, f1_score
# from datasets import set_caching_enabled
# set_caching_enabled(False)
from transformers import AutoTokenizer, DataCollatorForTokenClassification, TrainerCallback
from transformers import AutoModelForTokenClassification, TrainingArguments, Trainer

from .utils import convert_from_euler_to_huggingface, read_job_config, tokenize_and_align_labels


logger = logging.getLogger(__name__)
NAME2BASE = {
    "distilbert": "distilbert-base-uncased",
    "bert": "bert-base-chinese"
}


def get_configs():
    """
    获取配置
    """
    job_configs = read_job_config()
    logger.info("job_configs:{}".format(job_configs))
    # 读取input：
    for parameter in job_configs['input']:
        if parameter['name'] == 'train_file':
            # eg. "/data/conll2003/train.json"
            train_file = parameter['value']

        if parameter['name'] == 'val_file':
            # eg. "/data/conll2003/val.json"
            val_file = parameter['value']

        if parameter['name'] == 'label':
            # eg. "/data/conll2003/label.txt"
            label_file = parameter['value']

    # 读取output：
    for parameter in job_configs['output']:
        if parameter['name'] == 'export_dir':
            # eg. "/data/conll2003/debug_train/checkpoints"
            export_dir = parameter['value']
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)

    kwargs = job_configs.get('args', {})

    configs = {
        "label_col": "tags",
        "checkpoint": NAME2BASE[kwargs["model"]], # eg. "distilbert"
        "batch_size": kwargs["batch_size"], # eg. # 16
        "train_file": train_file,
        "val_file": val_file,
        "label_list": label_file,
        "output_dir": export_dir,
        "epochs": kwargs["epochs"], # eg. # 2
        "from_euler": kwargs.get("from_euler", True)
    }
    configs["cache_dir"] = os.path.dirname(export_dir)
    return configs


def get_datasets(configs):
    """
    获取数据集
    """
    if configs["from_euler"]:
        configs["train_file"], configs["label_list"] = convert_from_euler_to_huggingface(
            configs["train_file"], configs["label_list"], tag_name=configs["label_col"], change_label_file=True
        )
        configs["val_file"], configs["label_list"] = convert_from_euler_to_huggingface(
            configs["val_file"], configs["label_list"], tag_name=configs["label_col"], change_label_file=False
        )
    datasets = load_dataset(
        'json', 
        data_files={
            'train': configs["train_file"], 
            'validation': configs["val_file"]
        }, 
        field="data",
        cache_dir=os.path.dirname(configs["train_file"])
    )
    with open(configs["label_list"], "r") as f:
        label_list = f.readlines()
    return datasets, label_list


def compute_metrics(p, label_list):
    """
    计算模型指标
    """
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = {}
    for metric in [accuracy_score, precision_score, recall_score, f1_score]:
        results[metric.__name__] = metric(true_labels, true_predictions)
    return results


def inference(trainer, tokenized_datasets, label_list):
    """
    模型预测
    """
    predictions, labels, _ = trainer.predict(tokenized_datasets["validation"])
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = {}
    for metric in [accuracy_score, precision_score, recall_score, f1_score]:
        results[metric.__name__] = metric(true_labels, true_predictions)
    return results


class LogCallback(TrainerCallback):
    "A callback that prints some messages"
    def on_train_begin(self, args, state, control, **kwargs):
        logger.info("*"*10 + "start training" + "*"*10)

    def on_epoch_begin(self, args, state, control, **kwargs):
        logger.info ("Start training on epoch-{}\n".format(state.epoch + 1))

    def on_epoch_end(self, args, state, control, **kwargs):
        logger.info ("End training on epoch-{0}\n".format(state.epoch))

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        logger.info ("End evaluating on epoch-{}\n".format(state.epoch))
        logger.info("metrics: {}".format(metrics))


def main(configs):
    """
    主要逻辑
    """
    datasets, label_list = get_datasets(configs)
    logger.info("load pretrained AutoTokenizer")
    tokenizer = AutoTokenizer.from_pretrained(configs["checkpoint"], cache_dir="/data/model_pretrained_weights/huggingface")
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
    tokenized_datasets = datasets.map(
        partial(tokenize_and_align_labels, tokenizer=tokenizer, label_col=configs["label_col"]), 
        batched=True
    )
    logger.info("load pretrained AutoModelForTokenClassification")
    model = AutoModelForTokenClassification.from_pretrained(
        configs["checkpoint"], num_labels=len(label_list), cache_dir="/data/model_pretrained_weights/huggingface"
    )
    args = TrainingArguments(
        configs["output_dir"],
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=configs["batch_size"],
        per_device_eval_batch_size=configs["batch_size"],
        num_train_epochs=configs["epochs"],
        weight_decay=0.01,
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        disable_tqdm=True,
        report_to=None,
        metric_for_best_model="loss" # 默认loss为评价模型的指标
    )
    logger.info("log level-{}".format(args.get_process_log_level()))
    logger.info("using device-{}".format(args.device))

    data_collator = DataCollatorForTokenClassification(tokenizer)
    trainer = Trainer(
        model,
        args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, label_list=label_list),
        callbacks=[LogCallback]
    )
    trainer.train()

    logger.info("*"*10 + "start evaluating" + "*"*10)
    trainer.evaluate()

    best_checkpoint = trainer.state.best_model_checkpoint
    logger.info("best_checkpoint: {}".format(best_checkpoint))
    logger.info("loading from best_checkpoint")
    trainer.model = AutoModelForTokenClassification.from_pretrained(
        best_checkpoint, num_labels=len(label_list), cache_dir=configs["cache_dir"]
    ).to(args.device)
    
    logger.info("*"*10 + "start inference" + "*"*10)
    results = inference(trainer, tokenized_datasets, label_list)
    logger.info("final inference metric:{}".format(results))
    

if __name__ == "__main__":
    configs = get_configs()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
        datefmt='%Y-%m-%d %A %H:%M:%S'
    )
    logger.info("final configs:{}".format(configs))
    main(configs)
