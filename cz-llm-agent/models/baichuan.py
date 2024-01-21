# TODO(@zyw): 

from models.base import BaseLLM


class Baichuan2(BaseLLM): 

    def __init__(self, model_name_or_path: str): 
        pass


if __name__ == '__main__': 
    llm = Baichuan2("/path/to/model")
    messages = [
        {"role": "system", "content": "..."}, 
        {"role": "user", "content": "..."}, 
    ]
    functions = [{}]
    llm.change_messages(messages)
    output = llm.parse(functions=functions)
    print(output)