export PYTHONPATH=./
python preprocess/preprocess_toolllama_data.py \
    --tool_data_dir data/answer_chinese/test \
    --method DFS_woFilter_w2 \
    --output_file prep/answer/test_dfs_chinese.json