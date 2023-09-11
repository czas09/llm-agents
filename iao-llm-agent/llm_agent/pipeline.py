import json
import os
import os.path
import random
import re
from typing import List, Dict, Tuple, Any, Optional
import time

from loguru import logger
from pydantic import BaseModel
from termcolor import colored

from llm_agent.base_environ import BaseEnviron
from llm_agent.api_executor import get_api_response
from models.openai_api import ChatGPT
from models.toolllama import ToolLLaMA
from models.retriever import ToolRetriever
# from search_methods.cot import SingleChain
from search_methods.dfs import DFSChain
from models.utils import replace_llama_with_condense
from utils import standardize, change_name, get_toolkits, check_toolkits


# TODO(@zyw): 梳理项目源码中的各种数据结构
# Task = Tuple[str, str, int, Dict, Any, str, List[List[str, str]]]
# class Task_(BaseModel): 
#     search_method: str = "DFS"
#     model: str = "gpt-3.5-turbo"
#     query_id: int = 0
#     user_query: Dict = None
#     args: Any = None
#     answer_dir: str = None
#     tool_descriptions: List[List[str, str]]


class APIEnviron(BaseEnviron): 
    def __init__(self, user_query, tool_descriptions, retriever, args, process_id): 
        self.tool_root_dir = args.tool_root_dir
        # self.api_customization = args.api_customization
        self.max_observation_length = args.max_observation_length
        self.observ_compress_method = args.observ_compress_method
        self.retriever = retriever
        self.top_k = args.retrieval_top_k
        self.process_id = process_id
        self.tool_names = []
        self.category_names = []
        self.query_content = user_query["query"]
        self.functions = []
        self.api_name_reflect = {}    #

        if self.retriever is not None: 
            tools_for_this_query = self._retrieve_tools(self.query_content, self.top_k, self.tool_root_dir)
            tools_for_this_query = self._fetch_api_details(tools_for_this_query)
            tool_descriptions = self._build_tool_description(tools_for_this_query)
        else:
            tools_for_this_query = self._fetch_api_details(user_query)

        # TODO(@zyw)
        for index, api_doc in enumerate(tools_for_this_query["api_list"]):
            standard_tool_name = tool_descriptions[index][0]
            openai_format_function, category_name, api_name = self._to_openai_format(api_doc, standard_tool_name)
            self.functions.append(openai_format_function)

            self.api_name_reflect[openai_format_function["name"]] = api_name
            self.tool_names.append(standard_tool_name)
            self.category_names.append(category_name)

        # TODO(@zyw): Finish作为候选工具之一
        # ToolBench 项目中专门定义了“Finish”工具，用于处理终止和重启的情况
        finish_func = {
            "name": "Finish", 
            "description": "如果你认为你已经得到了可以回答任务的结果，请调用此函数来提供最终答案。或者，如果你意识到在当前状态下无法继续进行任务，请调用此函数来重新开始。请记住：你必须在尝试结束时始终调用此函数，而用户将只看到最终答案的部分，因此它应包含足够的信息。",
            "parameters": {
                "type": "object", 
                "properties": {
                    "return_type": {
                        "type": "string", 
                        "enum": [
                            "give_answer", 
                            "give_up_and_restart"
                        ],
                    },
                    "final_answer": {
                        "type": "string",
                        "description": "你想给到用户的最终答案。如果“return_type”等于“give_answer”，那么你就应该提供这个“final_answer”字段", 
                    }
                }, 
                "required": ["return_type"],
            }
        }
        self.functions.append(finish_func)
        self.CALL_MAX_TIME = 3
        # TODO(@zyw): 系统prompt中的任务描述字符串
        self.task_description = f'''您应该使用函数来处理用户提出的实时查询。请记住：
1.任务结束时始终调用“Finish”函数。最终答案应包含足够的信息以供用户查看。如果您无法处理任务，或者发现函数调用总是失败（这说明该函数当前无效），请使用函数“Finish->give_up_and_restart”。
2.不要使用原始工具名称，只使用子功能的名称。
您可以访问以下工具：'''

        # 在任务描述（task_description）的后面添加工具描述文本（tool_description）
        for standardize_tool_name, tool_description in tool_descriptions: 
            tool_description = tool_description[:512].replace('\n', '').strip()
            if tool_description == "": 
                tool_description = None
            self.task_description += "{}.{}: {}\n".format(index+1, standardize_tool_name, tool_description)
        
        self.success = 0

    def _build_tool_description(self, retrieved_tools): 
        toolkits = get_toolkits(self.tool_root_dir)
        retrieved_tools = [standardize(tool["tool_name"]) for tool in retrieved_tools["api_list"]]
        tools = check_toolkits(retrieved_tools, toolkits)
        tool_descriptions = [[tool.standard_tool_name, tool.description] for tool in tools]
        return tool_descriptions
        
    def _retrieve_tools(self, query: str, top_k: int, tool_root_dir: str): 
        retrieved_tools = self.retriever.retrieving(query, top_k=top_k)
        tools_for_this_query = {"api_list": []}
        for tool_dict in retrieved_tools: 
            if len(tools_for_this_query["api_list"]) == top_k: 
                break
            category_name = tool_dict["category_name"]
            tool_name = tool_dict["tool_name"]
            api_name = tool_dict["api_name"]
            if os.path.exists(tool_root_dir): 
                if os.path.exists(os.path.join(tool_root_dir, category_name)): 
                    if os.path.exists(os.path.join(tool_root_dir, category_name, tool_name+".json")): 
                        tools_for_this_query["api_list"].append({
                            "category_name": category_name,
                            "tool_name": tool_name,
                            "api_name": api_name
                        })
        return tools_for_this_query
    
    def _fetch_api_details(self, tools): 
        tools_with_details = {"api_list": []}
        for tool in tools["api_list"]: 
            category_name = tool["category_name"]
            tool_name = standardize(tool["tool_name"])
            api_name = change_name(standardize(tool["api_name"]))
            with open(os.path.join(self.tool_root_dir, category_name, tool_name+".json"), 'r', encoding='utf-8') as f: 
                tool_doc_json = json.load(f)
            append_flag = False
            api_dict_names = []
            for api_details in tool_doc_json["api_list"]: 
                api_dict_names.append(api_details["name"])
                api_name_in_doc = change_name(standardize(api_details["name"]))
                if api_name_in_doc != api_name: 
                    continue
                api_details_new = dict()
                api_details_new["category_name"] = category_name
                api_details_new["tool_name"] = tool_doc_json["tool_name"]
                api_details_new["api_name"] = api_name
                api_details_new["api_description"] = api_details["description"]
                api_details_new["required_parameters"] = api_details["required_parameters"]
                api_details_new["optional_parameters"] = api_details["optional_parameters"]
                tools_with_details["api_list"].append(api_details_new)
                append_flag = True
                break
            if not append_flag: 
                print(api_name, api_dict_names)
        return tools_with_details
    
    def _to_openai_format(self, api_doc, standard_tool_name): 
        description_max_length = 256
        openai_schema = {
            "name": "", 
            "description": "", 
            "parameters": {
                "type": "object", 
                "properties": {}, 
                "required": [], 
                "optional": [], 
            }
        }
        type_mappings = {
            "NUMBER": "integer", 
            "STRING": "string", 
            "BOOLEAN": "boolean", 
        }

        api_name = change_name(standardize(api_doc["api_name"]))
        openai_schema["name"] = "{}_for_{}".format(api_name, standard_tool_name)[-64:]
        openai_schema["description"] = "这是“{}”工具的子功能，你可以使用这个工具。".format(standard_tool_name)

        if api_doc["api_description"].strip() != "": 
            truncated_description = api_doc["api_description"].strip().replace(api_doc["api_name"], openai_schema["name"])[:description_max_length]
            openai_schema["description"] = "这个功能的描述文本如下：“{}”".format(truncated_description)
        
        if "required_parameters" in api_doc.keys() and len(api_doc["required_parameters"]) > 0: 
            # 必需参数
            for param in api_doc["required_parameters"]: 
                param_name = change_name(standardize(param["name"]))
                if param["type"] in type_mappings: 
                    param_type = type_mappings[param["type"]]
                else: 
                    param_type = "string"
                prompt_for_param = {
                    "type": param_type, 
                    "description": param["description"][:description_max_length], 
                }
                default_value = param["default"]
                if len(str(default_value)) != 0: 
                    prompt_for_param["example_value"] = default_value
                openai_schema["parameters"]["properties"][param_name] = prompt_for_param
                openai_schema["parameters"]["required"].append(param_name)
            
            # 可选参数
            for param in api_doc["optional_parameters"]: 
                param_name = change_name(standardize(param["name"]))
                if param["type"] in type_mappings: 
                    param_type = type_mappings[param["type"]]
                else: 
                    param_type = "string"
                default_value = param["default"]
                if len(str(default_value)) != 0: 
                    prompt_for_param["example_value"] = default_value
                openai_schema["parameters"]["properties"][param_name] = prompt_for_param
                openai_schema["parameters"]["optional"].append(param_name)

        return openai_schema, api_doc["category_name"], api_name

    def check_success(self):
        return self.success

    def to_json(self):
        return {}

    def restart(self):
        pass

    def get_score(self):
        return 0.0

    def execute(self, **args): 
        observation, status = self._execute(**args)
        if len(observation) > self.max_observation_length: 
            observation = observation[:self.max_observation_length] + "..."
        return observation, status
    
    def _execute(self, action_name="", action_input=""): 
        """Need to return an observation string and status code:
            0 means normal response
            1 means there is no corresponding api name
            2 means there is an error in the input
            3 represents the end of the generation and the final answer appears
            4 means that the model decides to pruning by itself
            5 represents api call timeout
            6 for 404
            7 means not subscribed
            8 represents unauthorized
            9 represents too many requests
            10 stands for rate limit
            11 message contains "error" field
            12 error sending request
        """
        if action_name == "Finish":
            try:
                json_data = json.loads(action_input, strict=False)
            except:
                json_data = {}
                if '"return_type": "' in action_input:
                    if '"return_type": "give_answer"' in action_input:
                        return_type = "give_answer"
                    elif '"return_type": "give_up_and_restart"' in action_input:
                        return_type = "give_up_and_restart"
                    else:
                        return_type = action_input[action_input.find('"return_type": "')+len('"return_type": "'):action_input.find('",')]
                    json_data["return_type"] = return_type
                if '"final_answer": "' in action_input:
                    final_answer = action_input[action_input.find('"final_answer": "')+len('"final_answer": "'):]
                    json_data["final_answer"] = final_answer

            if "return_type" not in json_data.keys():
                return "{error:\"must have \"return_type\"\"}", 2
            
            if json_data["return_type"] == "give_up_and_restart":
                return "{\"response\":\"chose to give up and restart\"}",4
            elif json_data["return_type"] == "give_answer":
                if "final_answer" not in json_data.keys():
                    return "{error:\"must have \"final_answer\"\"}", 2
                self.success = 1    # succesfully return final_answer
                return "{\"response\":\"successfully giving the final answer.\"}", 3
            else:
                return "{error:\"\"return_type\" is not a valid choice\"}", 2
            
        else:
            for index, function in enumerate(self.functions):
                if function["name"].endswith(action_name):
                    api_name = self.api_name_reflect[function["name"]]
                    payload = {
                        "category_name": self.category_names[index], 
                        "tool_name": self.tool_names[index], 
                        "api_name": api_name, 
                        "tool_input": action_input, 
                        "strip": self.observ_compress_method, 
                    }
                    if self.process_id == 0:
                        print(colored(f"query to {self.category_names[index]}-->{self.tool_names[index]}-->{action_name}", color="yellow"))

                    # 在这里实际执行API调用
                    response = get_api_response(payload)

                    # 1 Hallucinating function names
                    # 4 means that the model decides to pruning by itself
                    # 5 represents api call timeout
                    # 6 for 404
                    # 7 means not subscribed
                    # 8 represents unauthorized
                    # 9 represents too many requests
                    # 10 stands for rate limit
                    # 11 message contains "error" field
                    # 12 error sending request
                    if response["error"] == "API not working error...":
                        status_code = 6
                    elif response["error"] == "Unauthorized error...":
                        status_code = 7
                    elif response["error"] == "Unsubscribed error...":
                        status_code = 8
                    elif response["error"] == "Too many requests error...":
                        status_code = 9
                    elif response["error"] == "Rate limit per minute error...":
                        print("Reach api calling limit per minute, sleeping...")
                        time.sleep(10)
                        status_code = 10
                    elif response["error"] == "Message error...":
                        status_code = 11
                    else:
                        status_code = 0
                    return json.dumps(response), status_code
                    # except Exception as e:
                    #     return json.dumps({"error": f"Timeout error...{e}", "response": ""}), 5
            return json.dumps({"error": f"No such function name: {action_name}", "response": ""}), 1


class Pipeline: 

    def __init__(self, 
                 args, 
                 process_id: int = 0): 
        self.args = args
        self.use_retrieval = args.use_retrieval
        self.process_id = process_id
        self.model_name = args.model_name
        self.model_path = args.model_path
        self.openai_key = args.openai_key
        self.is_lora = args.is_lora
        self.lora_path = args.lora_path
        self.tool_corpus_path = args.tool_corpus_path
        self.retrieval_model_path = args.retrieval_model_path
        self.search_method = args.search_method
        self.query_file_path = args.query_file_path
        self.tool_root_dir = args.tool_root_dir
        self.answer_dir = args.answer_dir
        self.model = self._get_model()
        self.task_list = self._generate_task_list()

    def _get_model(self) -> str: 
        if self.model_name == "toolllama": 
            # ratio = 4 means the sequence length is expanded by 4, remember to change the model_max_length to 8192 (2048 * ratio) for ratio = 4
            replace_llama_with_condense(ratio=4)
            if self.is_lora: 
                # model = ToolLLaMALoRA(base_name_or_path=args.model_path, model_name_or_path=args.lora_path)
                raise NotImplementedError
            else: 
                model = ToolLLaMA(model_name_or_path=self.model_path)
        else:    # gpt-3.5-turbo或davinci
            model = self.model_name
        
        return model
    
    def _get_retriever(self) -> ToolRetriever: 
        return ToolRetriever(
            tool_corpus_path=self.tool_corpus_path, embed_model_path=self.retrieval_model_path)

    def _generate_task_list(self) -> list: 
        args = self.args
        query_file_path = self.query_file_path
        answer_dir = self.answer_dir
        if not os.path.exists(answer_dir):
            os.mkdir(answer_dir)
        search_method = self.search_method
        model = self.model
        
        with open(query_file_path, 'r', encoding='utf-8') as f: 
            user_queries = json.load(f)
        
        toolkits = get_toolkits(self.tool_root_dir)
        task_list = []
        for query_id, user_query in enumerate(user_queries): 
            query_id = user_query.get("query_id", query_id)
            if "api_list" in user_query:    # 用户主动提供的工具
                user_provide_tools = [standardize(api["tool_name"]) for api in user_query["api_list"]]
                tools = check_toolkits(user_provide_tools, toolkits)
                if not tools: 
                    continue
                tool_descriptions = [[tool.standard_tool_name, tool.description] for tool in tools]
            else: 
                tool_descriptions = None
            single_task = (search_method, model, query_id, user_query, args, answer_dir, tool_descriptions)
            task_list.append(single_task)

        return task_list
    
    def _construct_chain(self, llm, openai_key, search_method, api_environ, process_id, single_chain_max_step, max_query_count): 

        if llm == "chatgpt_function": 
            model = "gpt-3.5-turbo-16k-0613"
            llm = ChatGPT(model=model, openai_key=openai_key)
        elif llm == "davinci":    # TODO(@zyw)
            # model = "text-davinci-003"
            # llm = Davinci(model=model, openai_key=openai_key)
            raise NotImplementedError
        else: 
            model = llm    # toolllama
            llm = model
        
        if search_method.startswith("CoT"): 
            # pass_at = int(search_method.split("@")[-1])
            # chain = SingleChain(llm=llm, api_environ=api_environ, process_id=process_id)
            raise NotImplementedError
        elif search_method.startswith("DFS"): 
            pattern = r".+_w(\d+)"
            re_result = re.match(pattern, search_method)
            assert re_result != None
            width = int(re_result.group(1))
            with_filter = True
            if "woFilter" in search_method:
                with_filter = False
            chain = DFSChain(llm=llm, env=api_environ, process_id=process_id)
            result = chain.start(
                single_chain_max_step=single_chain_max_step, 
                tree_beam_size=width,    # DFS_woFilter_w2 -> 2
                max_query_count=max_query_count, 
                answer=1, 
                with_filter=with_filter
            )
        else: 
            print("invalid search method!")
            raise NotImplementedError
        
        return chain, result
    
    def _run_single_task(self, single_task, retriever: ToolRetriever, process_id: int = 0): 
        search_method, model, query_id, user_query, args, answer_dir, tool_descriptions = single_task
        parts = answer_dir.split("/")
        os.makedirs("/".join(parts[:-1]), exist_ok=True)
        os.makedirs("/".join(parts), exist_ok=True)
        answer_json_path = os.path.join(answer_dir, "{}_{}.json".format(query_id, search_method))
        if os.path.exists(answer_json_path): 
            return 
        
        api_environ = APIEnviron(user_query, tool_descriptions, retriever, args, process_id=process_id)

        query_content = user_query["query"]

        if process_id == 0:
            logger.info(f"[process({process_id})]now playing {query_content}, with {len(api_environ.functions)} APIs", "green")
        
        # ======================================================================
        # 根据search_method执行ReAct推理流程，生成原始数据
        # ======================================================================
        chain, result = self._construct_chain(
            llm=model, 
            openai_key=self.openai_key, 
            search_method=search_method,    # DFS_woFilter_w2
            api_environ=api_environ, 
            process_id=process_id, 
            single_chain_max_step=12, 
            max_query_count=200, 
        )

        # ======================================================================
        # 保存原始数据
        # 具体内容及格式可参考 data/answer/...json
        # ======================================================================
        if answer_dir is not None: 
            data = chain.to_json(answer=True, process=True)
            data["answer_generation"]["query"] = query_content
            with open(answer_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
            success = data["answer_generation"]["valid_data"] and "give_answer" in data["answer_generation"]["final_answer"]
            logger.info(f"[process({process_id})] valid={success}")
        return result

    
    def run(self): 
        task_list = self.task_list
        random.seed(1027)
        random.shuffle(task_list)
        logger.info("当前任务总数：{}".format(len(task_list)))

        undo_tasks = []
        for task in task_list: 
            answer_dir = task[-2]
            query_id = task[2]
            answer_json_path = os.path.join(answer_dir, "{}_{}.json".format(query_id, self.search_method))
            if not os.path.exists(answer_json_path): 
                undo_tasks.append(task)
        logger.info("当前未完成任务数量：{}".format(len(undo_tasks)))

        retriever = self._get_retriever() if self.use_retrieval else None

        for task_id, task in enumerate(task_list): 
            logger.info("进程[{}] 当前处理任务 {}/{}（任务原始序号：{}）".format(self.process_id, task_id, len(task_list), task[2]))
            _ = self._run_single_task(task, retriever, process_id=self.process_id)