from typing import Dict

from sentence_transformers import SentenceTransformer, util
import pandas as pd
from loguru import logger

from utils import process_tool_corpus, standardize, standardize_category, change_name


class ToolRetriever: 
    
    def __init__(self, tool_corpus_path: str, embed_model_path: str): 
        self.tool_corpus_path = tool_corpus_path
        self.embed_model_path = embed_model_path
        self.tool_corpus, self.corpus2tool_mappings = self._build_tool_corpus()
        self.embed_model = self._load_embed_model()
        self.tool_corpus_embeds = self._get_tool_corpus_embeds()

    def _build_tool_corpus(self): 
        corpus_df = pd.read_csv(self.tool_corpus_path, sep='\t')
        tool_corpus, corpus2tool_mappings = process_tool_corpus(corpus_df)
        tool_corpus = [tool_corpus[doc_id] for doc_id in list(tool_corpus.keys())]
        return tool_corpus, corpus2tool_mappings
    
    def _load_embed_model(self): 
        embed_model = SentenceTransformer(self.embed_model_path)
        return embed_model
    
    def _get_tool_corpus_embeds(self): 
        tool_corpus_embeds = self.embed_model.encode(self.tool_corpus, convert_to_tensor=True)
        return tool_corpus_embeds
    
    def retrieving(self, query: str, top_k: int = 5, excluded_tools: Dict[str, str] = {}): 
        query_embed = self.embed_model.encode(query, convert_to_tensor=True)
        results = util.semantic_search(query_embed, self.tool_corpus_embeds, top_k=10*top_k, score_function=util.cos_sim)
        retrieved_tools = []
        for result in results[0]: 
            category_name, tool_name, api_name = self.corpus2tool_mappings[self.tool_corpus[result['corpus_id']]].split('\t') 
            category_name = standardize_category(category_name)
            tool_name = standardize(tool_name)
            api_name = change_name(standardize(api_name))
            if category_name in excluded_tools:
                if tool_name in excluded_tools[category_name]:
                    top_k += 1
                    continue
            tmp_dict = {
                "category_name": category_name, 
                "tool_name": tool_name, 
                "api_name": api_name
            }
            retrieved_tools.append(tmp_dict)
        return retrieved_tools
