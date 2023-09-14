# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# 源码参考来源：
# tatsu-lab@stanford_alpaca (Alpaca模型的工作)
# fastchat (Vicuna模型的工作)
# toolbench (toolllama的工作)

from dataclasses import dataclass, field
import json
import os
import pathlib
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
import numpy as np

from tool_conversation import SeparatorStyle
from train.llama_condense_monkey_patch import replace_llama_with_condense


@dataclass
class ModelArguments: 
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")


@dataclass
class DataArguments: 
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    conv_template: str = field(
        default=None, metadata={"help": "Template used to format the training data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments): 
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    source_model_max_length: int = field(
        default=2048, 
        metadata={
            "help": "Original maximum sequence length. Sequences will be right padded (and possibly truncated)."
        }, 
    )
    model_max_length: int = field(
        default=8192, 
        metadata={
            "help": "Expanded maximum sequence length. Sequences will be right padded (and possibly truncated)."
        }
    )


local_rank = None


def make_supervised_data_module(
        tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict: 
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (

    )


def train(): 
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    if training_args.source_model_max_length < training_args.model_max_length: 
        condense_ratio = int(training_args.model_max_length / training_args.source_model_max_length)
        # ratio = N means the sequence length is expanded by N, remember to change the model_max_length to 8192 (2048 * ratio) for ratio = 4
        replace_llama_with_condense(ratio=condense_ratio)
    
    local_rank = training_args.local_rank
    tokenzier = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, 
        cache_dir=training_args.cache_dir, 
        model_max_length=data_args.model_max_length, 
        padding_side="right", 
        use_fast=False, 
    )
    tokenizer.pad_token = tokenizer.unk_token

    data_module = make