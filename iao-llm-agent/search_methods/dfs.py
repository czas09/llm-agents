from copy import deepcopy
import json
import random
import re
from typing import List

from search_methods.base import BaseSearchMethod
from tree.tree import Tree, TreeNode
from prompts.react_prompts import (
    FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION_ZH, 
    FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION, 
    FORMAT_INSTRUCTIONS_USER_FUNCTION_ZH, 
    FORMAT_INSTRUCTIONS_USER_FUNCTION, 
)
from prompts.tree_search_prompts import DIVERSITY_PROMPT_ZH, DIVERSITY_PROMPT
from models.base import BaseLLM
from llm_agent.base_environ import BaseEnviron
from rank.rank_candidates import rank2_subfix, sum_based_rankn


class DFSChain(BaseSearchMethod): 
    
    def __init__(self, llm, env, process_id=0): 
        super(DFSChain, self).__init__(llm, env, process_id)
        self.llm: BaseLLM = llm
        self.env = env
        self.prompt_lang = self.env.prompt_lang
        self.process_id = process_id
        self.restart()

    def restart(self): 
        self.status = 0    # 1 for terminal
        self.terminal_nodes: List[TreeNode] = []
        self.give_up_nodes: List[TreeNode] = []
        self.now_expand_num = 0
        self.query_count = 0
        self.total_tokens = 0

    def dfs(self, now_node: TreeNode, single_chain_max_step, tree_beam_size, max_query_count, answer, with_filter=True): 
        """Returns the number of grids to go back. When a child node of a node generates a final answer or give up, it should go back a few more grids
        In a sense, the larger this value is, the more diverse it is, and it is GreedySearch@n when it is enlarged to infinity.
        """

        # this two value declares the rate to go back, Algo degrades to CoT when the value=Inf
        final_answer_back_length = 2
        prune_back_length = 2

        now_node.expand_num = self.now_expand_num    # 在本次DFS迭代中，当前节点的被访问次数等于当前扩张次数
        self.now_expand_num += 1    # 记一次迭代
        # 边界条件：
        # 1.当前DFS迭代达到最大深度
        # 2.当前节点被判定为被剪枝状态(is_pruned)
        # 3.当前节点被判定为终点(is_terminal)，也就是得到了最终答案
        if now_node.get_depth() >= single_chain_max_step or now_node.pruned or now_node.is_terminal: 
            if now_node.is_terminal:    # final answer
                self.status = 1
                self.terminal_nodes.append(now_node)
                return final_answer_back_length
            else: 
                now_node.pruned = True
                if now_node.observation_code == 4: 
                    self.give_up_nodes.append(now_node)
                    return prune_back_length
                else: 
                    return 1
        
        next_tree_split_nodes = []
        for index in range(tree_beam_size): 
            temp_current_node: TreeNode = now_node

            """If a node have children now, We will prompt the model to generate different nodes than all the existing nodes"""
            # 边界条件：
            # 有孩子节点，说明在之前的迭代中存在失败的尝试，现在属于回溯到之前的某个节点
            delete_former_diversity_message = False
            diversity_message = None
            if len(temp_current_node.children) > 0: 
                former_candidates_description = ""
                json_list = []
                for child_node in temp_current_node.children: 
                    temp_child_node: TreeNode = child_node

                    # temp_child_node 非终点；非“Action Input”节点；有子节点
                    while not temp_child_node.is_terminal and temp_child_node.node_type != "Action Input" and len(temp_child_node.children) > 0: 
                    # while not temp_node.is_terminal and temp_node.node_type != "Action Input" and len(temp_node.children) > 0:
                        temp_child_node = temp_child_node.children[0]
                        # temp_node = temp_node.children[0]
                    if temp_child_node.node_type == "Action Input": 
                        json_obj = {
                            "name": temp_child_node.father.description, 
                            "arguments": temp_child_node.description, 
                            "function_output": temp_child_node.observation, 
                            "mento-carlo-action-value": temp_child_node.compute_weight(), 
                        }
                        json_list.append(json_obj)
                
                if len(json_list) > 0: 
                    former_candidates_description += f"{json.dumps(json_list, indent=2, ensure_ascii=False)}\n"
                    if temp_current_node.observation != "": 
                        if self.prompt_lang == 'zh': 
                            former_candidates_description += f"此外，你之前的观察结果(observation)是：{temp_current_node.observation}\n"
                        else:    # en
                            former_candidates_description += f"again, your former observation: {temp_current_node.observation}\n"
                    
                    if self.prompt_lang == 'zh': 
                        diverse_prompt = DIVERSITY_PROMPT_ZH
                    else:    # en
                        diverse_prompt = DIVERSITY_PROMPT
                    diverse_prompt = diverse_prompt.replace(
                        "{previous_candidate}", 
                        former_candidates_description)
                    diversity_message = {
                        "role": "user", 
                        "content": diverse_prompt
                    }
                    temp_current_node.messages.append(diversity_message)
                    delete_former_diversity_message = True
            
            now_depth = temp_current_node.get_depth() // 3
            agent_block_ids = []
            self.llm.change_messages(temp_current_node.messages)
            new_message, error_code, total_token_usage = self.llm.parse(self.env.functions, process_id=self.process_id)
            self.query_count += 1
            self.total_tokens += total_token_usage
            if self.query_count >= max_query_count:    # a big return value will cause the Algo to exit
                return 100000

            # We need to exclude the diversity_message, because it will influence child nodes
            if delete_former_diversity_message:
                temp_current_node.messages[-1]["valid"] = False
            
            # ==================================================================
            # parse nodes from OpenAI-message like CoT method
            # 解析大模型输出结果（role=assistant），构建新节点（3种）
            # Thought -> content
            # Action -> function_call
            # Action Input -> function_call arguments
            # ==================================================================
            assert new_message["role"] == "assistant"
            
            # 1.首先基于content内容构建Thought节点
            # （一般是出错和重启的时候会创建Thought，因为Function call成功的话，模型返回结果不会带content）
            if "content" in new_message.keys() and new_message["content"] != None: 
                temp_new_node = TreeNode()
                temp_new_node.node_type = "Thought"
                temp_new_node.description = new_message["content"]
                new_env = deepcopy(temp_current_node.env)
                new_env.retriever = None
                temp_new_node.env = new_env
                temp_new_node.is_terminal = new_env.check_success() != 0
                temp_new_node.messages = deepcopy(temp_current_node.messages)
                temp_new_node.father = temp_current_node
                temp_current_node.children.append(temp_new_node)
                temp_new_node.print(self.process_id)
                temp_current_node = temp_new_node
                
                if error_code != 0: 
                    temp_current_node.observation_code = error_code
                    temp_current_node.pruned = True
            
            if "function_call" in new_message.keys(): 
                # 2.接着基于function_call内容构建Action节点
                function_name = new_message["function_call"]["name"]
                temp_new_node = TreeNode()
                temp_new_node.node_type = "Action"
                temp_new_node.description = function_name
                new_env = deepcopy(temp_current_node.env)
                new_env.retriever = None
                temp_new_node.env = new_env
                temp_new_node.is_terminal = new_env.check_success() != 0
                temp_new_node.messages = deepcopy(temp_current_node.messages)
                temp_new_node.father = temp_current_node
                temp_current_node.children.append(temp_new_node)
                temp_new_node.print(self.process_id)
                temp_current_node = temp_new_node

                # 3.基于function_call arguments 内容构建Action Input节点
                function_input = new_message["function_call"]["arguments"]
                temp_new_node = TreeNode()
                temp_new_node.node_type = "Action Input"
                temp_new_node.description = function_input
                new_env = deepcopy(temp_current_node.env)
                new_env.retriever = None
                observation, status = new_env.execute(
                    action_name=temp_current_node.description, 
                    action_input=function_input)
                temp_new_node.observation = observation
                temp_new_node.observation_code = status
                temp_new_node.env = new_env
                temp_new_node.is_terminal = new_env.check_success() != 0
                temp_new_node.messages = deepcopy(temp_current_node.messages)
                temp_new_node.father = temp_current_node
                temp_current_node.children.append(temp_new_node)
                temp_new_node.print(self.process_id)
                temp_current_node = temp_new_node

                if status != 0:          # 出错，状态码参见APIEnviron()中的注释
                    if status == 4:      # means that the model decides to pruning by itself
                        temp_current_node.pruned = True
                    elif status == 1:    # means there is no corresponding api name (hallucination)
                        assert "function_call" in new_message.keys()
                        new_message["function_call"]["name"] = "invalid_hallucination_function_name"
                    elif status == 3:    # represents the end of the generation and the final answer appears
                        temp_current_node.is_terminal = True
                        temp_current_node.make_finish(final_answer_back_length)
            
            temp_current_node.messages.append(new_message)
            if temp_current_node.node_type == "Action Input": 
                temp_current_node.messages.append({
                    "role": "function", 
                    "name": new_message["function_call"]["name"], 
                    "content": temp_current_node.observation})
                
            return_value = None
            if not with_filter:    # DFSDT
                result = self.dfs(
                    temp_current_node, 
                    single_chain_max_step, tree_beam_size, max_query_count, answer, with_filter)
                if len(self.terminal_nodes) >= answer: 
                    return_value = 10000
                elif result > 1: 
                    return_value = result - 1
            else: 
                next_tree_split_nodes.append(temp_current_node)
            if return_value is not None: 
                return return_value

        # ==================================================
        # TODO(@zyw)
        # ==================================================
        # Sort the generated next_tree_split_nodes nodes when normal DFS
        if len(next_tree_split_nodes) > 1:
            # When using normal DFS, if we have many child nodes, we will refer to LLM to compare and choose the best one to expand first
            # remember, this operator will cost extra OpenAI calls.
            LLM_rank_args = {
                "functions": self.env.functions,
                "process_id": self.process_id,
                "task_description": self.env.task_description,
                "rank_func": rank2_subfix,
            }
            scores, rank_query_count, total_tokens = sum_based_rankn(
                self.llm, LLM_rank_args=LLM_rank_args, candidates=next_tree_split_nodes)
            self.query_count += rank_query_count
            self.total_tokens += total_tokens
            for score, node in zip(scores, next_tree_split_nodes):
                node.prior_score = score
            zip_value = list(
                zip(next_tree_split_nodes, range(len(next_tree_split_nodes))))
            zip_value.sort(
                key=lambda x: x[0].prior_score, reverse=True)  # 先做score高的
            next_tree_split_nodes, filtered_order = zip(*zip_value)
            # if self.process_id == 0:
            #     print(f"score={scores}, filtered order: {filtered_order}")

        '''
        Choose one to expand
        '''
        for i in range(len(next_tree_split_nodes)):
            result = self.dfs(
                next_tree_split_nodes[i], single_chain_max_step, tree_beam_size, max_query_count, answer)
            if len(self.terminal_nodes) >= answer: 
                return 10000
            elif result > 1: 
                now_node.make_finish(2)
                return result - 1

        return 1
    
    def start(self, single_chain_max_step, tree_beam_size, max_query_count, answer=1, with_filter=True): 
        """ single_chain_max_step: The maximum depth of the tree
            tree_beam_size: How many children nodes for one node are generated per layer
            answer = n means the Algo exits when find n "give_answer" nodes
            max_query_count: the Algo exits when OpenAI-query exists this value
            with_filter: This is the difference between normal DFS(with_filter=True) and DFSDT(with_filter=False). 
        """
        self.forward_args = locals()
        if "self" in self.forward_args.keys(): 
            self.forward_args.pop("self")
        self.tree = Tree()
        self.tree.root.node_type = "Action Input"
        self.tree.root.env = deepcopy(self.env)

        if self.prompt_lang == 'zh': 
            system_instruction = FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION_ZH
        else:    # en
            system_instruction = FORMAT_INSTRUCTIONS_SYSTEM_FUNCTION
        system_instruction = system_instruction.replace(
            "{task_description}", 
            self.env.task_description)
        self.tree.root.messages.append({
            "role": "system", 
            "content": system_instruction})

        if self.prompt_lang == 'zh': 
            user_input = FORMAT_INSTRUCTIONS_USER_FUNCTION_ZH
        else:    # en
            user_input = FORMAT_INSTRUCTIONS_USER_FUNCTION
        user_input = user_input.replace(
            "{input_description}", 
            self.env.query_content)
        self.tree.root.messages.append({
            "role": "user", 
            "content": user_input})
        
        return self.dfs(
            now_node=self.tree.root, 
            single_chain_max_step=single_chain_max_step, 
            tree_beam_size=tree_beam_size, 
            max_query_count=max_query_count, 
            answer=answer, 
            with_filter=with_filter)
    
    def to_json(self, answer=False, process=True):
        if process: 
            json_obj = {
                "win": self.status == 1, 
                "tree": self.tree.to_json_recursive(), 
                "forward_args": self.forward_args, 
                "compare_candidates": [], 
            }
            for node in self.terminal_nodes:
                if node.pruned == False:    # has answer
                    json_obj["compare_candidates"].append(
                        node.get_chain_result_from_this_node(use_messages=False))
        else:
            json_obj = {}

        if answer: 
            json_obj["answer_generation"] = {
                "valid_data": False, 
                "query_count": self.query_count, 
                "total_tokens": self.total_tokens, 
                "final_answer": "", 
                "finish_type": "give_answer", 
                "function": self.env.functions, 
                "chain": [], 
            }
            for node in self.terminal_nodes:
                if node.pruned == False:
                    json_obj["answer_generation"]["valid_data"] = True
                    json_obj["answer_generation"]["finish_type"] = "give_answer"
                    json_obj["answer_generation"]["final_answer"] = node.description
                    json_obj["answer_generation"]["train_messages"] = node.get_train_messages_from_this_node()
                    break
            # do not have final answer, look for give_up
            if json_obj["answer_generation"]["valid_data"] == False:
                if len(self.give_up_nodes) > 0:
                    random_pos = random.randint(0, len(self.give_up_nodes) - 1)
                    choose_give_up_nodes = self.give_up_nodes[random_pos]
                    json_obj["answer_generation"]["valid_data"] = True
                    json_obj["answer_generation"]["finish_type"] = "give_up"
                    json_obj["answer_generation"]["final_answer"] = choose_give_up_nodes.description
                    json_obj["answer_generation"]["train_messages"] = choose_give_up_nodes.get_train_messages_from_this_node()
        return json_obj