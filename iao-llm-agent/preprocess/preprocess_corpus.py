import argparse
import json
import os
from tqdm import tqdm

import pandas as pd
from sklearn.utils import shuffle

# 创建参数解析器并添加参数
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default="", required=True, help='')
parser.add_argument('--output_dir', type=str, default="", required=True, help='')
args = parser.parse_args()

input_dir = args.input_dir
output_dir = args.output_dir



documents_df.to_csv(output_dir + '/corpus.tsv', sep='\t', index=False)