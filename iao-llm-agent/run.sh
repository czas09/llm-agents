export OPENAI_API_BASE="https://key.langchain.com.cn/v1"
export OPENAI_API_PREFIX="ttps://key.langchain.com.cn"
export OPENAI_KEY="sk-kcfJcDXKztSEuMxaSqVjvuniMFIlz8HSr2xApuxivkNINiEc"
export PYTHONPATH=./

python3 run.py \
  --model_name chatgpt_function \
  --openai_key $OPENAI_KEY \
  --tool_root_dir data/toolenv/tools/ \
  --tool_corpus_path data/retrieval/corpus.tsv \
  --retrieval_model_path \
  --max_observation_length 1024 \
  --method DFS_woFilter_w2 \
  --input_query_file data/instruction/query_open_domain.json \
  --output_answer_file data/answer/test \
  --tool_corpus_path data/retrieval/corpus.tsv \
  --retrieval_model_path /home/xk/zyw/playground/models/m3e-base-20230608 \
  --retrieved_api_nums 2