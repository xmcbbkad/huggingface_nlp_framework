import os
import logging
from huggingface_deploy.nlp.token_classification_demo.classifier import TokenClassifer


logger = logging.getLogger(__name__)


class HuaziExtract():
    def __init__(self, file_name=None, **kwargs):
        super(HuaziExtract, self).__init__(file_name, **kwargs)
        tokenizer_dir = file_name
        model_path = os.path.join(file_name, kwargs.get("onnx_name", "best.onnx"))
        model="bert"
        label_list = kwargs.get("label_list", [])
        self.classifier = TokenClassifer(tokenizer_dir, model_path, model, label_list)

    def convert_input(self, data):
        """
        转换输入数据的格式

        Parameters
        ----------
        data : list[dict]
            输入的数据
        data[*].raw_str : str
            原始的字符串

        Returns
        -------
        classifier_input : dict
            输入的数据
        classifier_input.tokens : list[str]
            输入的句子列表
        """
        classifier_input = {"tokens": []}
        for item in data:
            classifier_input["tokens"].append(item["raw_str"])
        return classifier_input

    def predict(self, data, **kwargs):
        """
        通过tokenizer处理输入的字符串，并通过onnx对token进行分类

        Parameters
        ----------
        data : list[dict]
            输入的数据
        data[*].raw_str : str
            原始的字符串

        Returns
        -------
            即self.classifier的输出
        """
        classifier_input = self.convert_input(data)
        results = self.classifier.predict(classifier_input, **kwargs)
        return results

if __name__ == "__main__":
    model_path = ""
    kwargs = {
        "onnx_name": "best.onnx",
        "label_list": ["O", "B-MAIN", "I-MAIN"]
    }

    model = HuaziExtract(file_name=model_path, **kwargs)

    data = [
        {"raw_str": "这瓶雅诗兰黛面霜效果非常好，能让脸润润的，有光泽"}
    ]

    result = model.predict(data)
    logger.fatal("result:{}".format(result))
    print("result:{}".format(result))
