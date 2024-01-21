# ==============================================================================
# 中文提示词模板
# ==============================================================================
DIVERSITY_PROMPT_ZH = '''这不是你第一次尝试这个任务，之前的所有尝试都失败了。
在为这个状态生成我的思考(thought)之前，我会先向你展示你之前在这个状态下的操作(action)，然后你必须生成与它们都不同的操作(action)。以下是一些之前的操作(action)候选项：
{previous_candidate}
请记住，你现在处于一轮尝试的中间状态，你要先分析当前状态和之前的操作候选项，然后做出与之前的所有操作(action)都不同的操作(action)。'''

# ==============================================================================
# 英文提示词模板
# ==============================================================================
DIVERSITY_PROMPT = '''This is not the first time you try this task, all previous trails failed.
Before you generate my thought for this state, I will first show you your previous actions for this state, and then you must generate actions that is different from all of them. Here are some previous actions candidates:
{previous_candidate}
Remember you are now in the intermediate state of a trail, you will first analyze the now state and previous action candidates, then make actions that is different from all the previous.'''
