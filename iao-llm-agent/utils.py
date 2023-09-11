import json
import os
import os.path
import re
from typing import List, Dict

import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm
import torch


class Tool(BaseModel): 
    description: str
    standard_tool_name: str


def standardize_category(category):
    save_category = category.replace(" ", "_").replace(",", "_").replace("/", "_")
    while " " in save_category or "," in save_category:
        save_category = save_category.replace(" ", "_").replace(",", "_")
    save_category = save_category.replace("__", "_")
    return save_category


def standardize(string: str) -> str: 
    # 在string中匹配任何不是中文字符、小写字母、大写字母、数字或下划线的字符，并将其替换为下划线（_）
    res = re.compile("[^\\u4e00-\\u9fa5^a-z^A-Z^0-9^_]")
    string = res.sub("_", string)
    string = re.sub(r"(_)\1+","_", string).lower()
    
    # 去除string开头和结尾的下划线
    while True:
        if len(string) == 0:
            return string
        if string[0] == "_":
            string = string[1:]
        else:
            break
    while True:
        if len(string) == 0:
            return string
        if string[-1] == "_":
            string = string[:-1]
        else:
            break

    if string[0].isdigit():
        string = "get_" + string

    return string


def change_name(name):
    change_list = ["from", "class", "return", "false", "true", "id", "and"]
    if name in change_list:
        name = "is_" + name
    return name


def get_toolkits(tool_root_dir: str):
    tool_dir = os.path.join(tool_root_dir)
    toolkits = {}
    categories = os.listdir(tool_dir)
    for category in tqdm(categories): 
        file_path = os.path.join(tool_dir, category)
        if not os.path.isdir(file_path):
            continue
        for tool_doc_file in os.listdir(file_path):
            if not tool_doc_file.endswith(".json"):
                continue
            standard_tool_name = tool_doc_file.split(".")[0]    # TODO(@zyw): 工具的标准名称
            with open(os.path.join(tool_dir, category, tool_doc_file)) as f:
                tool_doc_json = json.load(f)
            origin_tool_name = standardize(tool_doc_json["tool_name"])       # TODO(@zyw): 工具的原始名称
            toolkits[origin_tool_name] = Tool(
                description=tool_doc_json["tool_description"], 
                standard_tool_name=standard_tool_name)
    return toolkits


def check_toolkits(user_provide_tools: List[str], toolkits: Dict[str, Tool]): 
    tools = []
    for tool_name in user_provide_tools: 
        if tool_name not in toolkits.keys(): 
            return False
        tools.append(toolkits[tool_name])
    return tools


def process_tool_corpus(corpus_df): 
    tool_corpus = dict()
    corpus2tool_mappings = dict()
    for entry in corpus_df.itertuples(): 
        doc_content = json.loads(entry.document_content)

        category_name = doc_content.get('category_name', '')
        tool_name = doc_content.get('tool_name', '')
        api_name = doc_content.get('api_name', '')
        api_description = doc_content.get('api_description', '')
        required_params = json.dumps(doc_content.get('required_parameters', ''))
        optional_params = json.dumps(doc_content.get('optional_parameters', ''))
        template_response = json.dumps(doc_content.get('template_response', ''))
        doc_str = "{}, {}, {}, {}, required_params: {}, optional_params: {}, return_schema: {}".format(
            category_name, tool_name, api_name, api_description, 
            required_params, optional_params, template_response)
        tool_corpus[entry.docid] = doc_str
        corpus2tool_mappings[doc_str] = "{}\t{}\t{}".format(category_name, tool_name, api_name)
    
    return tool_corpus, corpus2tool_mappings


def process_system_message(system_message: str, functions) -> str: 
    assert "同时给出一个函数调用来实际执行你的步骤。" in system_message
    # we find that following ReACT format and merging the thought node and function call node is easier for model to learn to integrate the action input json string in its prediction than learn to predict a json string directly.
    # 在开源模型（LLaMA、ToolLLaMA等）中采用以下ReAct格式，并且合并 thought 节点和 function call 节点
    # 与让模型自己生成 action input json串相比，合并后的效果更好
    system_message = system_message.replace("同时给出一个函数调用来实际执行你的步骤。", "同时给出一个函数调用来实际执行你的步骤。你的输出应该符合以下格式：\nThought:\nAction\nAction Input:\n")
    # add all the function dicts in the prompt.
    system_message = system_message + "\n特别地，你可以访问到以下的APIs: " + str(functions)
    return system_message


def get_gpu_memory(max_gpus=None):
    """Get available memory for each GPU."""
    gpu_memory = []
    num_gpus = (
        torch.cuda.device_count()
        if max_gpus is None
        else min(max_gpus, torch.cuda.device_count())
    )

    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return gpu_memory