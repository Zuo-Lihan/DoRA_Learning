# 1: 加载相关库
import importlib
import math
import re
import warnings
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D

from ..utils import PeftConfig, PeftType, transpose

# 2。==============================================
def is_bnb_available():
    return importlib.util.find_spec("bitsandbytes") is not None

if is_bnb_available():
    import bitsandbytes as bnb

''' 
importlib.util.find_spec:
    ~. find_spec 用于查找模块的规格即specification，它会返回一个ModuleSpec对象，包含了有关模块的各种信息，如果模块不可用则返回None。

bitsandbytes:
    ~. 适用于深度学习模型加速和优化的库。主要功能包括：
        1. 低精度运算：
            ~。支持半精度的浮点数（FP16）和混合精度训练。使用较低的精度有助于减少内存使用。
        2. 权重量化：
            ~。 支持对神经网络的权重进行量化，例如使用8位整数（INT8）来代替32位的浮点数（FP32），从而减少内存。
            ~。 量化可以在推理阶段显著加快模型的推理速度。
        3. 内存优化：
            ~。 提供了高效的内存管理策略，可以有效的避免内存碎片化的问题，提高内存使用效率。
        4. 计算加速：
            ~。 使用自定义的高效内核和优化算法，提高矩阵乘法、卷积操作等基础计算的速度。
'''

# 3。 ==================================================
@dataclass
class DoraConfig(PeftConfig):
    """
    This is the configuration class to store the configuration of a [`~peft.Lora`].

    Args:
        r (`int`): Lora attention dimension
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora to.
        lora_alpha (`float`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        merge_weights (`bool`):
            Whether to merge the weights of the Lora layers with the base transformer model in `eval` mode.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        enable_lora ( `List[bool]`): Used with `lora.MergedLinear`.
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
    """

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=None, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=None, metadata={"help": "Lora dropout"})
    dora_simple: bool = field(
        default=True, metadata={"help": "Whether to apply simple dora ver to save up GPU memory"}
    )
    Wdecompose_target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to only tune the magnitude part"
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    merge_weights: bool = field(
        default=False, metadata={"help": "Merge weights of the original model and the Lora model"}
    )
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    enable_lora: Optional[List[bool]] = field(default=None, metadata={"help": "Used with `lora.MergedLinear`."})
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )

    def __post_init__(self):
        self.peft_type = PeftType.DORA

'''
1。 在class前定义@dataclass，是为了简化数据类的定义，使其能够自动生成一些常用的内部方法，例如：
__init__, __repr__, __eq__等。
另外，使用@dataclass可以通过“field”函数为定义的字段添加元数据和帮助信息，非常有用。

2。 DoraConfig是一个从PeftConfig继承的类，是一个微调模型的相关配置信息：
    r: 
        用于Lora调控的秩的维数；
    target_modules:
        ~。Optional[]：表示这是一个可选的字段，可以是"None"或者指定的类型；
        ~。Union[List[str], str]：表示该字段可以是一个字符串列表或者是单个的字符串，单个的字符串可以是正则表达式匹配模型中的模块；
    lora_alpha：
        lora模型的调节参数，主要用于微调训练过程中，通常为1，控制前向传播过程中将LoRA权重应用于预训练权重的程度；
    lora_dropout：
        随机控制更新参数的几率；
    dora_simple：
        是否使用简单的dora模块，用于调节使用GPU资源的程度；
        dora_simple具体实现什么功能，后续再看；
    Wdecompose_target_modules:
        应用文章设计的decompose方法的目标模块；
    merge_weights:
        是否将原始的预训练模型与LoRA后的模型进行整合，整合方式有多种，例如加权平均；
        这样做，可以保留原始模型的一部分特性和信息，并结合Lora模型的改进部分；
    fan_in_fan_out:
        虽然 Transformer 模型本身已经内置了处理输入序列依赖关系的能力，但在特定情况下仍然可能需要 fan_in 和 fan_out 参数，主要是为了权重初始化、优化器调整、定制化需求以及性能优化等目的；
    enable_lora：
        是一个bool列表，控制对于每个lora模块加入原权重的添加方式，通常为线性；
    bias:
        偏置选项，None, All, 或者是 Lora_only;
    modules_to_save: 
        除了Lora模块需要被训练后保存，可能还包括一些其他模块，例如分类器classifier需要被训练然后保存；

3。__post_init__(self):
    是dataclass中的特殊方法，用于在对象初始化完成后执行任何需要的额外初始化操作，
    在python的dataclass中，他会在对象被创建后自动调用。
    
    self.peft_type = PeftType.DORA：
        这一行代码将对象的peft_type属性设置为PeftType.DORA；  
'''

# 4。========================================================== 模型模块，我将注释加在对应位置
class DoraModel(torch.nn.Module):
    """
    Creates Low Rank Adapter (Lora) model from a pretrained transformers model.

    Args:
        model ([`transformers.PreTrainedModel`]): The model to be adapted.
        config ([`LoraConfig`]): The configuration of the Lora model.

    Returns:
        `torch.nn.Module`: The Lora model.

    Example::

        >>> from transformers import AutoModelForSeq2SeqLM, LoraConfig >>> from peft import LoraModel, LoraConfig >>>
        config = LoraConfig(
            peft_type="LORA", task_type="SEQ_2_SEQ_LM", r=8, lora_alpha=32, target_modules=["q", "v"],
            lora_dropout=0.01, )
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base") >>> lora_model = LoraModel(config, model)

    **Attributes**:
        - **model** ([`transformers.PreTrainedModel`]) -- The model to be adapted.
        - **peft_config** ([`LoraConfig`]): The configuration of the Lora model.
    """

    def __init__(self, config, model):  # 初始化，config，model
        super().__init__()
        self.peft_config = config   # 定义self.peft_config
        self.model = model  # 定义self.model
        self._find_and_replace()    # self._find_and_replace()看后面
        mark_only_lora_as_trainable(self.model, self.peft_config.bias)  # mark_only_lora_as_trainable(self.model, self.peft_config.bias)
        self.forward = self.model.forward   # 定义self.forward

    def _find_and_replace(self):    # 定义_find_and_replace函数，
        loaded_in_8bit = getattr(self.model, "is_loaded_in_8bit", False)    # 检查self.model对象中是否有is_loaded_in_8bit的属性
        if loaded_in_8bit and not is_bnb_available():   # 如果有loaded_in_8bit，但是没有is_bnb_available()，那么需要加载bnb模块
            raise ImportError(
                "To use Lora with 8-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )
        is_target_modules_in_base_model = False # 定义一个标签变量，is_target_modules_in_base_model
        is_hf_device_map_available = hasattr(self.model, "hf_device_map")   # hf_device_map帮助定义哪些层或模块应当分配到哪些GPU上。
        kwargs = {  # 参数kwargs
            "r": self.peft_config.r,
            "lora_alpha": self.peft_config.lora_alpha,
            "lora_dropout": self.peft_config.lora_dropout,
            "fan_in_fan_out": self.peft_config.fan_in_fan_out,
            "merge_weights": (self.peft_config.merge_weights or self.peft_config.inference_mode)
                             and not is_hf_device_map_available,    # 逻辑表达意思是：只有在self.peft_config.merge_weights为true，或者self.peft_config.inference_mode为true时，并且is_hf_device_map_available为false时，才将merge_weights设置为True；
            "dora_simple": self.peft_config.dora_simple
        }
        key_list = [key for key, _ in self.model.named_modules()]   # 读出模型内部的named_modules()存储为key_list
        for key in key_list:
            if isinstance(self.peft_config.target_modules, str):    # 检查self.peft_config.target_modules是否是str类型；
                target_module_found = re.fullmatch(self.peft_config.target_modules, key)    # 全匹配，self.peft_config.target_modules是否与key匹配
            else:
                target_module_found = any(key.endswith(target_key) for target_key in self.peft_config.target_modules)

            if isinstance(self.peft_config.Wdecompose_target_modules, str):
                wdecompose_target_module_found = re.fullmatch(self.peft_config.Wdecompose_target_modules, key)
            elif self.peft_config.Wdecompose_target_modules == None:
                wdecompose_target_module_found = False
            else:
                wdecompose_target_module_found = any(
                    key.endswith(target_key) for target_key in self.peft_config.Wdecompose_target_modules)

            if target_module_found: # 如果找到目标模块
                if not is_target_modules_in_base_model: # 检查is_target_modules_in_base_model是否为False，如果为False则执行下面的代码
                    is_target_modules_in_base_model = True  # 置为True
                parent, target, target_name = self._get_submodules(key) # 调用self._get_submodules(key)
                bias = target.bias is not None  # target.bias属性存在且不为None，则返回bool值给bias，设为True
                if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):  # 检查是否可以loaded_in_8bit,并且target目标是否是bnb.nn.Linear8bitLt类型
                    kwargs.update(  # 更新关键字参数
                        {
                            "has_fp16_weights": target.state.has_fp16_weights,
                            "memory_efficient_backward": target.state.memory_efficient_backward,
                            "threshold": target.state.threshold,
                            "index": target.index,
                        }
                    )
                    if self.peft_config.enable_lora is None:    # 如果self.peft_config.enable_lora 为None
                        new_module = Linear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs) # 创建Linear8bitLt类型对象
                    else:   # 否则：
                        raise NotImplementedError

                elif isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:  # 否则在不能loaded_in_8bit条件下时，目标target模块是torch.nn.Linear类型，且peft_config.enable_lora为None时，执行下面程序：
                    new_module = Linear(target.in_features, target.out_features, bias=bias, **kwargs)   # 创建新模块new_module
                elif self.peft_config.enable_lora is not None:
                    raise NotImplementedError

                self._replace_module(parent, target_name, new_module, target)   # 调用self._replace_module函数，具体内容见后续

            elif wdecompose_target_module_found:    # 如果wdecompose_target_module_found，和当前key模块匹配则替换
                if not is_target_modules_in_base_model: # 如果is_target_modules_in_base_model为False，执行
                    is_target_modules_in_base_model = True
                parent, target, target_name = self._get_submodules(key) # 获取对应key的模块
                bias = target.bias is not None
                if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
                    kwargs.update(
                        {
                            "has_fp16_weights": target.state.has_fp16_weights,
                            "memory_efficient_backward": target.state.memory_efficient_backward,
                            "threshold": target.state.threshold,
                            "index": target.index,
                        }
                    )
                    if self.peft_config.enable_lora is None:
                        new_module = Linear8bitLt(target.in_features, target.out_features, bias=bias, **kwargs)
                    else:
                        raise NotImplementedError

                elif isinstance(target, torch.nn.Linear) and self.peft_config.enable_lora is None:
                    new_module = Linear(target.in_features, target.out_features, bias=bias, Wdecompose=True, **kwargs)
                elif self.peft_config.enable_lora is not None:
                    raise NotImplementedError
                self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {self.peft_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _get_submodules(self, key): # 获取对应key的模块
        parent = self.model.get_submodule(".".join(key.split(".")[:-1]))
        target_name = key.split(".")[-1]
        target = self.model.get_submodule(key)
        return parent, target, target_name

    def _replace_module(self, parent_module, child_name, new_module, old_module):   # 替换
        setattr(parent_module, child_name, new_module)  # setattr是一个内置函数，将parent_module中名为child_name的属性（即子模块）替换为new_module，
        new_module.weight = old_module.weight   # 再将old_module的权重赋值给new_module.

        #
        with torch.no_grad():
            magnitude = (torch.linalg.norm(new_module.weight.detach(), dim=1)).unsqueeze(1).detach()    # torch.linalg.norm 计算张量的指定维度上的范数
            new_module.weight_m_wdecomp.weight.copy_(magnitude) # 将magnitude复制到指定模块的指定部分

        if old_module.bias is not None:
            new_module.bias = old_module.bias
        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name or "weight_m_wdecomp" in name:
                module.to(old_module.weight.device)

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped module."""
        try:
            return super().__getattr__(name)  # defer to nn.Module's logic
        except AttributeError:
            return getattr(self.model, name)

    @property   # 将方法定义为属性
    def modules_to_save(self):
        return None

    def get_peft_config_as_dict(self, inference: bool = False): # 获取config_dict
        config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(self.peft_config).items()}
        if inference:
            config["inference_mode"] = True
        return config

    def _set_adapter_layers(self, enabled=True):    # 如果某个模块是LoraLayer类型的实例，如果enabled为true则将module.disable_adapters设置为False；如果enabled为False，则将module.disable_adapters设为True
        for module in self.model.modules():
            if isinstance(module, LoraLayer):
                module.disable_adapters = False if enabled else True

    def enable_adapter_layers(self):
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self):
        self._set_adapter_layers(enabled=False)


# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


# had to adapt it for `lora_only` to work
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n and "weight_m_wdecomp" not in n:    # 非lora和decompose模块就设置为不要求梯度
            p.requires_grad = False
        else:
            print(f"{n} is trainable")
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, LoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError


class LoraLayer:
    def __init__(
            self,
            r: int,
            lora_alpha: int,
            lora_dropout: float,
            merge_weights: bool,
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        # Mark the weight as unmerged
        self.merged = False
        self.merge_weights = merge_weights
        self.disable_adapters = False


class Linear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            fan_in_fan_out: bool = False,
            # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
            merge_weights: bool = True,
            Wdecompose: bool = False,
            dora_simple: bool = True,
            **kwargs,
    ):
        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        LoraLayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, merge_weights=merge_weights)

        self.weight_m_wdecomp = nn.Linear(1, out_features,
                                          bias=False)  # self.weight_m_wdecomp.weight # shape: out_features, 1

        self.fan_in_fan_out = fan_in_fan_out
        self.Wdecompose = Wdecompose  # whether to tune only the magnitude component of Wdecompose or not
        self.dora_simple = dora_simple  # whether to use dora simple to save up GPU memory
        if self.Wdecompose == False:
            if r > 0:
                self.lora_A = nn.Linear(in_features, r, bias=False)
                self.lora_B = nn.Linear(r, out_features, bias=False)
                self.scaling = self.lora_alpha / self.r
                # Freezing the pre-trained weight matrix

        self.weight.requires_grad = False
        self.reset_parameters()
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, "lora_A"):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)

    def train(self, mode: bool = True):
        nn.Linear.train(self, mode)
        if self.Wdecompose == False:
            self.lora_A.train(mode)
            self.lora_B.train(mode)
        self.weight_m_wdecomp.train(mode)

        if not mode and self.merge_weights and not self.merged: # self.merge_weights指示是否需要合并权重， self.merged 标记权重是否已经合并过
            # Merge the weights and mark it
            if self.Wdecompose:
                # 计算权重的归一化因子
                norm_scale = (self.weight_m_wdecomp.weight / (torch.linalg.norm(self.weight, dim=1)).unsqueeze(1))
                weight = norm_scale * self.weight
                self.weight.data.copy_(weight.detach())
            else:
                if self.r > 0:
                    new_weight_v = self.weight + transpose(self.lora_B.weight @ self.lora_A.weight,
                                                           fan_in_fan_out=self.fan_in_fan_out) * self.scaling
                    weight = (self.weight_m_wdecomp.weight / (torch.linalg.norm(new_weight_v, dim=1)).unsqueeze(
                        1)) * new_weight_v
                    self.weight.data.copy_(weight.detach())
            self.merged = True
        elif self.merge_weights and self.merged:
            raise NotImplementedError

    def eval(self):
        nn.Linear.eval(self)
        if self.Wdecompose == False:
            self.lora_A.eval()
            self.lora_B.eval()
        self.weight_m_wdecomp.eval()

    def forward(self, x: torch.Tensor):
        previous_dtype = self.weight.dtype

        if self.disable_adapters:
            raise NotImplementedError

        elif self.Wdecompose and not self.merged:

            norm_scale = self.weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(self.weight, dim=1))

            org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))

            result = org_result + (norm_scale - 1) * (
                F.linear(self.lora_dropout(x), transpose(self.weight, self.fan_in_fan_out)))

            if not self.bias is None:
                result += self.bias.view(1, -1).expand_as(result)

        elif self.r > 0 and not self.merged:

            new_weight_v = self.weight + (self.lora_B.weight @ self.lora_A.weight) * self.scaling

            if self.dora_simple:
                norm_scale = self.weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v, dim=1)).detach()
            else:
                norm_scale = self.weight_m_wdecomp.weight.view(-1) / (torch.linalg.norm(new_weight_v, dim=1))

            org_result = (F.linear(x, transpose(self.weight, self.fan_in_fan_out)))

            dropout_x = self.lora_dropout(x)

            result = org_result + (norm_scale - 1) * (F.linear(dropout_x, transpose(self.weight, self.fan_in_fan_out)))

            if not self.bias is None:
                result += self.bias.view(1, -1).expand_as(result)

            result += (norm_scale * (self.lora_B(self.lora_A(dropout_x.to(self.lora_A.weight.dtype))))) * self.scaling

        else:
            result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

        if result.dtype != previous_dtype:
            result = result.to(previous_dtype)

        return result


class MergedLinear(nn.Linear, LoraLayer):
    # Lora implemented in a dense layer
    def __init__(
            self,
            in_features: int,
            out_features: int,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            enable_lora: List[bool] = [False],
            fan_in_fan_out: bool = False,
            merge_weights: bool = True,
            **kwargs,
    ):
        raise NotImplementedError


if is_bnb_available():
    class Linear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
                self,
                in_features,
                out_features,
                r: int = 0,
                lora_alpha: int = 1,
                lora_dropout: float = 0.0,
                Wdecompose: bool = False,
                **kwargs,
        ):
            raise NotImplementedError


    class MergedLinear8bitLt(bnb.nn.Linear8bitLt, LoraLayer):
        # Lora implemented in a dense layer
        def __init__(
                self,
                in_features: int,
                out_features: int,
                r: int = 0,
                lora_alpha: int = 1,
                lora_dropout: float = 0.0,
                enable_lora: List[bool] = [False],
                **kwargs,
        ):
            raise NotImplementedError