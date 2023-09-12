#!/bin/bash

export PYTHONPATH=./
export INPUT_DIR="toolenv_chinese/tools"
export OUTPUT_DIR="data/retrieval/G1"

# data_preprocess.py will process files and output to the specified directory
python preprocess/preprocess_corpus.py \
  --input_dir $INPUT_DIR \
  --output_dir $OUTPUT_DIR
