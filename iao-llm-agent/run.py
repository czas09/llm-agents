'''
Close-domain QA Pipeline
'''

import argparse

from llm_agent.pipeline import Pipeline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default="toolllama", required=False, help='模型名称，可以是chatgpt、davinci或toolllama')
    parser.add_argument('--openai_key', type=str, default="", required=False, help='openai key for chatgpt_function or davinci model')
    parser.add_argument('--model_path', type=str, default="your_model_path/", required=False, help='')
    parser.add_argument('--tool_root_dir', type=str, default="your_tools_path/", required=True, help='')
    parser.add_argument("--is_lora", action="store_true", help="是否加载LoRA模型")
    parser.add_argument('--lora_path', type=str, default="your_lora_path if lora", required=False, help='')
    parser.add_argument('--use_retrieval', action="store_true", help='开放域场景')
    parser.add_argument('--tool_corpus_path', type=str, default="your_retrival_corpus_path/", required=False, help='')
    parser.add_argument('--retrieval_model_path', type=str, default="your_model_path/", required=False, help='')
    parser.add_argument('--retrieval_top_k', type=int, default=5, required=False, help='')
    parser.add_argument('--max_observation_length', type=int, default=1024, required=False, help='maximum observation length')
    parser.add_argument('--observ_compress_method', type=str, default="truncate", choices=["truncate", "filter", "random"], required=False, help='observation compress method')
    parser.add_argument('--search_method', type=str, default="CoT@1", required=False, help='method for answer generation: CoT@n, Reflexion@n, BFS, DFS, UCT_vote')
    parser.add_argument('--query_file_path', type=str, default="", required=False, help='input path')
    parser.add_argument('--answer_dir', type=str, default="",required=False, help='output path')
    # parser.add_argument('--toolbench_key', type=str, default="",required=False, help='your toolbench key to request rapidapi service')
    # parser.add_argument('--rapidapi_key', type=str, default="",required=False, help='your rapidapi key to request rapidapi service')
    # parser.add_argument('--use_rapidapi_key', action="store_true", help="To use customized rapidapi service or not.")
    # parser.add_argument('--api_customization', action="store_true", help="To use customized api or not.")

    args = parser.parse_args()

    pipeline = Pipeline(args)
    pipeline.run()
