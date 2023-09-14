
# TODO(@zyw): 训练哪一个模型？主要微调要用base模型！
# LLaMA、Llama-2
# 百川
# 千问
# 书生
# ChatGLM (只有chat模型)


# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.
# 节省内存




# Need to call this before importing transformers.
# 需要在导入 Transformers 之前调用这个方法。
from train.llama_flash_attn_monkey_patch import (
    replace_llama_attn_with_flash_attn,
)

replace_llama_attn_with_flash_attn()

from toolbench.train.train import train

if __name__ == "__main__":
    train()
