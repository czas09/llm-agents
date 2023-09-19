import time
import json

import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored

from models.base import BaseLLM


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(openai_key, messages, functions=None, function_call=None, key_pos=None, model="gpt-3.5-turbo-0613", stop=None, process_id=0, **args): 
    use_messages = []
    for message in messages: 
        if not ("valid" in message.keys() and message["valid"] == False): 
            use_messages.append(message)
    
    request_body = {
        "model": model, 
        "messages": use_messages, 
        "max_tokens": 1024, 
        "frequency_penalty": 0,
        "presence_penalty": 0,
        **args
    }
    if stop is not None:
        request_body.update({"stop": stop})
    if functions is not None:
        request_body.update({"functions": functions})
    if function_call is not None:
        request_body.update({"function_call": function_call})

    try:
        # if model == "gpt-3.5-turbo-16k-0613":
        if model == "gpt-3.5-turbo-0613":
            openai.api_key = openai_key
        else:
            raise NotImplementedError
        openai_response = openai.ChatCompletion.create(**request_body)
        openai_response = json.loads(str(openai_response))
        return openai_response 
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"OpenAI calling Exception: {e}")
        return e


class ChatGPT(BaseLLM): 
    
    def __init__(self, model="gpt-3.5-turbo-0613", openai_key=""): 
        self.model=model
        self.conversation_history = []
        self.openai_key = openai_key
        self.time = time.time()
        self.TRY_TIME = 6

    def add_message(self, message):
        self.conversation_history.append(message)

    def change_messages(self, messages):
        self.conversation_history = messages

    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red", 
            "user": "green", 
            "assistant": "blue", 
            "function": "magenta", 
        }
        print("before_print" + "*" * 50)
        for message in self.conversation_history:
            print_obj = f"{message['role']}: {message['content']} "
            if "function_call" in message.keys():
                print_obj = print_obj + f"function_call: {message['function_call']}"
            print_obj += ""
            print(colored(print_obj, role_to_color[message["role"]],))
        print("end_print" + "*" * 50)
    
    def parse(self, functions, process_id, key_pos=None, **args): 
        conversation_history = self.conversation_history
        for occasion in range(self.TRY_TIME): 
            if occasion != 0: 
                time.sleep(15)
            if functions != []: 
                openai_response = chat_completion_request(
                    self.openai_key, 
                    conversation_history, 
                    functions=functions, 
                    process_id=process_id, 
                    key_pos=key_pos, 
                    **args)
            else: 
                openai_response = chat_completion_request(
                    self.openai_key, 
                    conversation_history, 
                    process_id=process_id, 
                    key_pos=key_pos, 
                    **args)
                
            try: 
                total_token_usage = openai_response["usage"]["total_tokens"]
                message = openai_response["choices"][0]["message"]
                if process_id == 0: 
                    print(f"[process({process_id})]total tokens: {openai_response['usage']['total_tokens']}")
                if "function_call" in message.keys() and "." in message["function_call"]["name"]:
                    message["function_call"]["name"] = message["function_call"]["name"].split(".")[-1]
                return message, 0, total_token_usage
            except BaseException as e: 
                print(f"[process({process_id})]Parsing Exception: {repr(e)}. Try again.")
                if openai_response is not None:
                    print(f"[process({process_id})]OpenAI return: {openai_response}")

        return {
            "role": "assistant", 
            "content": str(openai_response)
        }, -1, 0



if __name__ == "__main__": 
    llm = ChatGPT(openai_key="sk-kcfJcDXKztSEuMxaSqVjvuniMFIlz8HSr2xApuxivkNINiEc")
#     prompt = '''下面这句英文可能有语病，能不能把语病都改掉？
# If you think you get the result which can answer the task, call this function to give the final answer. Or, if you think you can't handle the task from this status, call this function to restart. Remember: you should ALWAYS call this function at the end of your try, and the final answer is the ONLY part that will be showed to user, so final answer should contain enough information.
# 没语病的形式：
# '''
#     messages = [
#         {"role": "system", "content": ""}, 
#         {"role": "user", "content": prompt}, 
#     ]

    messages = [
        {'role': 'system', 'content': '''你是AutoGPT，你可使用很多工具（功能）来完成以下任务。\n首先，我会向你提供任务描述，然后你就开始执行你的任务。\n在每一个步骤中，你都要给出你的思考(Thought)来分析当前状态并决定下一步该做什么，同时给出一个函数调用来实际执行你的步骤。你的输出应该符合以下格式：\nThought:\nAction\nAction Input:\n在函数调用之后，你将获得调用结果，这时你将进入一个新的状态。\n
然后你要分析当前所处的状态，然后决定下一步该做什么...\n在经过多轮（思考(Thought)-调用(call)）循环之后，你将最终完成任务，然后你可以给出你的最终答案。\n请记住：\n1.状态的改变是不可逆转的，你不能返回到前面任何一个状态，如果你想重新开始任务的话，请说“我放弃当前尝试，任务重启”。\n2.你所有的思考(thought)都应当简短，最多不超过5个句子。\n3.你可以执行多次尝试，所以如果你计划持续尝试某些条件，你可以在每轮尝试中执行其中一个条件。\n让我们开始吧！\n任务描述：您应该使用函数来处理用户提出的实时查询。请记住：\n1.任务结束时始终调用“Finish”函数。最终答案应包含足够的信息以供用户查看。如果您无法处理任务，或者发现函数调用总是失败（这说明该函数当前无效），请使用函数“Finish->give_up_and_restart”。\n2.不要使用原始工具名称，只使用子功能的名称。\n'''}, 
{'role': 'user', 'content': '请给我“hello world”字符串。\n开始！\n'}
]
    functions = [{'name': 'hello_world', 'description': 'To get \'hello world\'.', 'parameters':{"type": "object", "properties": {}}}]
    llm.change_messages(messages=messages)
    response, error_code, token_usage = llm.parse(functions=functions, process_id=0)
    print(response)