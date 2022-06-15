import os
import glob
import logging
import tarfile
from pathlib import Path

from transformers.onnx import export
from transformers.models.distilbert import DistilBertOnnxConfig
from transformers.models.bert import BertOnnxConfig
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig

from .utils import read_job_config


logger = logging.getLogger(__name__)


def _get_onnx_config(args: dict):
    """
    获取用于导出onnx的配置，在onnx_export(args: dict)中被调用

    Parameters
    ----------
    args : dict
        同onnx_export(args: dict)的输入
    
    Returns
    -------
    onnx_config : OnnxConfig
        transformer中的一个类，对onnx动态输入的配置

    """
    config = AutoConfig.from_pretrained(args["model_ckpt"], cache_dir=args["cache_dir"])
    if args["model"] == "distilbert":
        onnx_config = DistilBertOnnxConfig(config)
    elif args["model"] == "bert":
        onnx_config = BertOnnxConfig(config)
    else:
        raise Exception("{} onnx config not implemented".format(args["model"]))
    return onnx_config


def onnx_export(args: dict) -> None:
    """
    导出onnx主程序

    Parameters
    ----------
    args : dict
        输入的配置
    args.model : str
        所使用的模型，可选项["distilbert", "bert"]，这与训练的模型要保持一致
    args.model_ckpt : str
        训练时模型结果的输出路径，由get_configs()自动挑选最底层目录
    args.onnx_path : str
        onnx文件的导出路径
    args.num_labels : int
        label的数量
    args.from_euler : bool
        默认为True，输入的数据结果是否是euler格式的
    args.cache_dir : str
        存放缓存的路径，由get_configs()自动生成

    Returns
    -------
    None
        结果存在args["onnx_path"]当中
    """
    logger.info("*"*10 + "start load_model" + "*"*10)
    base_model = AutoModelForTokenClassification.from_pretrained(
        args["model_ckpt"], num_labels=args["num_labels"], cache_dir=args["cache_dir"]
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args["model_ckpt"], cache_dir=args["cache_dir"]
    )
    logger.info("*"*10 + "start get_onnx_config" + "*"*10)
    onnx_config = _get_onnx_config(args)
    logger.info("*"*10 + "start export" + "*"*10)
    onnx_inputs, onnx_outputs = export(
        tokenizer, base_model, onnx_config, onnx_config.default_onnx_opset, args["onnx_path"]
    )
    logger.info("onnx_inputs: {}".format(onnx_inputs))
    logger.info("onnx_outputs: {}".format(onnx_outputs))


def get_configs() -> dict:
    """
    获取export配置，对从环境变量中读取到的配置进行一些结构转化

    Parameters
    ----------
    None

    Returns
    -------
    configs : dict
        onnx_export(args: dict)的输入
    """
    job_configs = read_job_config()
    logger.info("job_configs:{}".format(job_configs))
    # 读取input：
    for parameter in job_configs['input']:
        if parameter['name'] == 'model_ckpt_dir':
            # eg. "/data/conll2003/debug_train/checkpoints"
            model_ckpt = parameter['value']
        if parameter['name'] == 'label':
            # eg. "/data/conll2003/label.txt"
            label_file = parameter['value']
    with open(label_file, "r") as f:
        label_list = [ele.strip() for ele in f.readlines() if len(ele.strip()) > 0]
    
    # 读取output：
    for parameter in job_configs['output']:
        if parameter['name'] == 'onnx_path':
            # eg. "/data/conll2003/debug_train/onnx/best.onnx"
            onnx_path = parameter['value']
    
    # 转化结构
    kwargs = job_configs.get('args', {})
    configs = {
        "model": kwargs.get("model", "distilbert"),
        "model_ckpt" : model_ckpt,
        "onnx_path" : Path(onnx_path),
        "num_labels" : len(label_list),
        "from_euler": kwargs.get("from_euler", True)
    }
    configs["cache_dir"] = os.path.dirname(model_ckpt)
    checkpoints = glob.glob(os.path.join(configs["model_ckpt"], "*"))
    checkpoints = [checkpoint for checkpoint in checkpoints if os.path.basename(checkpoint).startswith('checkpoint')]
    if len(checkpoints) == 1:
        configs["model_ckpt"] = checkpoints[0]
    elif len(checkpoints) > 1:
        checkpoints = sorted(checkpoints, key=lambda x:int(x.split('-')[-1]))
        configs["model_ckpt"] = checkpoints[-1]
    else:
        pass
    os.makedirs(os.path.dirname(configs["onnx_path"]), exist_ok=True)
    if configs["from_euler"]:
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
        configs["num_labels"] = len(label_list)
    return configs


def make_tarfile(configs: dict) -> None:
    """
    打包用于部署的文件

    Parameters
    ----------
    configs : dict
        同onnx_export(args: dict)的输入
    
    Returns
    -------
    None
        打包euler上部署的tar包，存在args["onnx_path"]同级目录下。这个包可以改个名字传到oss上，然后
        填到euler的Model Path下
    """
    tar_path = os.path.join(os.path.dirname(configs["onnx_path"]), "results.tar.gz")
    with tarfile.open(tar_path, "w:gz") as tar:
        # 添加onnx文件
        tar.add(configs["onnx_path"], arcname=os.path.basename(configs["onnx_path"]))
        for filename in [
            "config.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json", "tokenizer.json"
        ]:
            tar.add(os.path.join(configs["model_ckpt"], filename), arcname=filename)
    logger.info("tarfile saved at {}".format(tar_path))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(filename)s : %(levelname)s  %(message)s',
        datefmt='%Y-%m-%d %A %H:%M:%S'
    )
    configs = get_configs()
    logger.info("final configs:{}".format(configs))
    onnx_export(configs)
    make_tarfile(configs)

    
