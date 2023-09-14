import os
import os.path
import json

import pandas as pd

input_dir = "data/toolenv_chinese/tools"

# print(os.listdir(input_dir))

corpus = []
cnt = 1
for entry1 in os.listdir(input_dir): 
    for entry2 in os.listdir(os.path.join(input_dir, entry1)): 
        if entry2.endswith(".json"): 
            # print(os.path.join(input_dir, entry1, entry2))
            api_dict = {}
            with open(os.path.join(input_dir, entry1, entry2), "r", encoding="utf-8") as f: 
                data = json.load(f)
                # print(data["api_list"])
            for api in data["api_list"]: 
                api_dict["category_name"] = entry1
                api_dict["tool_name"] = data["tool_name"]
                api_dict["api_name"] = api["name"]
                api_dict["api_description"] = api["description"]
                api_dict["required_parameters"] = api["required_parameters"]
                api_dict["optional_parameters"] = api["optional_parameters"]
                api_dict["method"] = api["method"]
                # print(api_dict)
                # print(json.dumps(api_dict, ensure_ascii=False))
                # str = json.dumps(api_dict, ensure_ascii=False).replace("\"", "\"\"").replace("'", "\"")
                corpus.append([cnt, json.dumps(api_dict, ensure_ascii=False)])
                cnt += 1

corpus_df = pd.DataFrame(corpus, columns=['docid', 'document_content'])
corpus_df.to_csv('./corpus.tsv', sep='\t', index=False)