import copy
from contextlib import contextmanager
from types import MethodType

import torch
from peft import PeftModel
from torch import nn
from transformers import PreTrainedModel

from surogate.core.model.model_info import ModelInfo
from surogate.core.model.registry import ModelTemplate
from surogate.utils.logger import get_logger
from surogate.utils.utils import deep_getattr

logger = get_logger()


def patch_get_input_embeddings(model, embedding_keys: str):
    def get_input_embeddings(self) -> nn.Module:
        return deep_getattr(model, embedding_keys)

    model.get_input_embeddings = MethodType(get_input_embeddings, model)


def patch_awq_compat(model_info: ModelInfo):
    if model_info.quant_method != 'awq':
        return

    try:
        # compat transformers>=4.50 (autoawq)
        from transformers.quantizers.quantizer_awq import AwqQuantizer
        from transformers.integrations import get_keys_to_not_convert
        _process_model_before_weight_loading = AwqQuantizer._process_model_before_weight_loading

        def _new_process_model_before_weight_loading(self, model, *args, **kwargs):
            modules_to_not_convert = self.quantization_config.modules_to_not_convert
            if modules_to_not_convert is not None:
                self.quantization_config.modules_to_not_convert = list(
                    modules_to_not_convert) + get_keys_to_not_convert(model)
            return _process_model_before_weight_loading(self, model, *args, **kwargs)

        AwqQuantizer._process_model_before_weight_loading = _new_process_model_before_weight_loading
    except Exception:
        pass

def get_lm_head_model(model, model_template: ModelTemplate = None, lm_heads=None):
    if isinstance(model, PeftModel):
        model = model.model
    model_template = model_template or model.model_template
    lm_heads = lm_heads or ['lm_head']
    llm_prefix_list = getattr(model_template.model_arch, 'language_model', None)
    prefix_list = []
    if llm_prefix_list:
        prefix_list = llm_prefix_list[0].split('.')

    current_model = model
    for prefix in prefix_list:
        current_model = getattr(current_model, prefix)
        for lm_head in lm_heads:
            if hasattr(current_model, lm_head):
                return current_model
    return model


@contextmanager
def patch_automodel(model_info, model_template, automodel_class, **kwargs):
    from_pretrained = PreTrainedModel.from_pretrained.__func__

    @classmethod
    def _new_from_pretrained(cls, *args, **kwargs):
        if 'AutoAWQFor' in automodel_class.__name__:
            kwargs.pop('use_cache', None)
        if model_info.quant_method == 'gptq':
            cls.main_input_name = 'input_ids'
        if hasattr(cls, '_tp_plan'):  # fix tp_plan
            cls._tp_plan = cls._tp_plan or {}
        return from_pretrained(cls, *args, **kwargs)

    PreTrainedModel.from_pretrained = _new_from_pretrained

    try:
        yield
    finally:
        PreTrainedModel.from_pretrained = classmethod(from_pretrained)


def patch_getattr(obj_cls, item_name: str):
    if hasattr(obj_cls, '_patch'):  # avoid double patch
        return

    def __new_getattr__(self, key: str):
        try:
            return super(self.__class__, self).__getattr__(key)
        except AttributeError:
            if item_name in dir(self):
                item = getattr(self, item_name)
                return getattr(item, key)
            raise

    obj_cls.__getattr__ = __new_getattr__
    obj_cls._patch = True

