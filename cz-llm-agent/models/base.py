class BaseLLM: 

    def __init__(self): 
        pass

    def add_message(self): 
        raise NotImplementedError
    
    def change_messages(self, messages):
        raise NotImplementedError

    def display_conversation(self, detailed=False): 
        raise NotImplementedError
    
    def parse(self): 
        raise NotImplementedError