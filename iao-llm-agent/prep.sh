export PYTHONPATH=./
python preprocess/preprocess_toolllama_data.py \
    --tool_data_dir data/answer/test \
    --method DFS_woFilter_w2 \
    --output_file prep/answer/test_dfs.json