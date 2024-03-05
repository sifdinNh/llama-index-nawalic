from typing import Sequence, Optional

from llama_index.bridge.pydantic import BaseModel,Field
from llama_index.schema import NodeWithScore
from llama_index.postprocessor import SentenceTransformerRerank
from llama_index import QueryBundle

class NawalicReranker(BaseModel):
    nodes : Sequence[NodeWithScore] = Field(description="The retrieved  base nodes for reranking")
    query : str = Field(description="the original user-specified query string")
    top_k : int = Field(description="number of top k results to return",default=2)
    reranking_model : str = Field(description="model's name to use for reranking", default="BAAI/bge-reranker-base")


    def rerank(self):
        reranker = SentenceTransformerRerank(top_n=self.top_k, model=self.reranking_model)
        reranked_nodes = reranker.postprocess_nodes(
            self.nodes,
            query_bundle=QueryBundle(
                self.query
            ),
        )
        return reranked_nodes