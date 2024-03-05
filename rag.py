import os

from llama_index.bridge.pydantic import BaseModel, Field
from llama_index.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.retrievers import BM25Retriever

from indexing import NawalicVectorIndex
from retreivers import HybridRetriever
from reranker import NawalicReranker
from llama_index.retrievers import QueryFusionRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index import ServiceContext, PromptTemplate,SelectorPromptTemplate
from llama_index.llms import OpenAI



QUERY_GEN_PROMPT = (
    "You are a helpful assistant that generates multiple search queries based on a "
    "single input query. the search queries must be in arabic."
    "Generate {num_queries} search queries, one on each line, "
    "related to the following input query:\n"
    "Query: {query}\n"
    "Queries:\n"
)

SUMMARY_PROMPT_TEMPLATE = """
Context information from multiple sources is below.\n
---------------------\n
{context_str}\n
---------------------\n
Given the information from multiple sources and not prior knowledge, answer the query.\n
The answer must be in arabic \n
include all metadata information in the answer\n
Query: {query_str}\n
Answer: 
"""
#التاريخ و الفصل و الدور و المضبطة

class NawalicRAG(BaseModel):
    ocr_data_dir : str = Field(description="Dir path for processed ocr data")
    document_metadata_dir : str = Field(description="Dir path for extracted metadata")
    vector_data_dir : str = Field(description="Directory path for vector store data")
    top_k : int = Field(description="number of top k results to return")


    def _get_vector_index(self):
        return NawalicVectorIndex(
            ocr_data_dir = self.ocr_data_dir,
            document_metadata_dir = self.document_metadata_dir,
            vector_data_dir = self.vector_data_dir,
        ).get_vector_index()
    

    def _retrieve(self, query):
#        index = self._get_vector_index()
#        vector_retriever = VectorIndexRetriever(
#            index=index,
#            similarity_top_k=self.top_k,
#        )
        retriever = self._hybrid_retriever()
        
        retrieved_nodes = retriever.retrieve(query)
#        reranker = NawalicReranker(nodes=retrieved_nodes,query=query)
#        return reranker.rerank()
        return retrieved_nodes
    
    def _hybrid_retriever(self):
        index = self._get_vector_index()
        vector_retriever = VectorIndexRetriever(
            index=index,
            similarity_top_k=3,
        )
        nodes = index.docstore.docs.values()
        bm25_retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=3)
        retriever = QueryFusionRetriever(
            [vector_retriever, bm25_retriever],
            similarity_top_k=2,
            num_queries=1,  # set this to 1 to disable query generation
            mode="reciprocal_rerank",
            use_async=True,
            verbose=True,
            query_gen_prompt=QUERY_GEN_PROMPT, 
        )
#        HybridRetriever(vector_retriever=vector_retriever,bm25_retriever=bm25_retriever)
        return retriever
    
    def query(self,query):
        retriever = self._hybrid_retriever()
        llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo", max_tokens=1056)
        service_context = ServiceContext.from_defaults(llm=llm)
        summary_template = PromptTemplate(template=SUMMARY_PROMPT_TEMPLATE)
        query_engine = RetrieverQueryEngine.from_args(retriever=retriever,service_context=service_context,response_mode="tree_summarize",summary_template=summary_template)
        response = query_engine.query(query)
        return response

