import  os, json
import re
from typing import List
from pathlib import Path
from llama_index import ServiceContext, SimpleDirectoryReader, KnowledgeGraphIndex, VectorStoreIndex, StorageContext, load_index_from_storage, QueryBundle
from llama_index.llms import OpenAI
from llama_index.extractors import SummaryExtractor, KeywordExtractor, QuestionsAnsweredExtractor, EntityExtractor
from llama_index.ingestion import IngestionPipeline
from llama_index.node_parser import SentenceSplitter, TokenTextSplitter
from llama_index.bridge.pydantic import BaseModel, Field
from llama_index import set_global_service_context
from llama_index.program.openai_program import OpenAIPydanticProgram
from llama_index.graph_stores import SimpleGraphStore
from llama_index.extractors import PydanticProgramExtractor
from llama_index.retrievers import KGTableRetriever, BM25Retriever
from llama_index.indices.vector_store.retrievers import VectorIndexRetriever
from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters
from llama_index.postprocessor import SentenceTransformerRerank


os.environ["OPENAI_API_KEY"] = "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"

SUMMARY_PROMPT_TEMPLATE = """
هاهو محتوى النص:
{context_str}

لخص المواضيع والكيانات الرئيسية للنص
Summary:
"""

DEFAULT_QUESTION_GEN_TMPL = """\
هاهو محتوى النص::
{context_str}

وبالنظر إلى المعلومات السياقية, \
أنشئ {num_questions} من الأسئلة التي يمكن أن يقدمها السياق \
إجابات محددة من غير المرجح أن تجدها في أي مكان آخر.

قد يتم توفير ملخصات عالية المستوى للسياق المحيط \
أيضًا. حاول استخدام هذه الملخصات لتوليد أسئلة أفضل \
أن هذا السياق يمكن الإجابة.

"""


def extract_number(filename):
    # Extract the number from the filename and return it as an integer
    match = re.search(r'\d+', filename.stem)
    return int(match.group()) if match else 0




class NodeMetadata(BaseModel):
    """Node metadata."""

    arabic_person_names: List[str] = Field(
        ..., description="Unique arabic Person names in this text chunk."
    )
    arabic_summary: str = Field(
        ..., description="A concise summary of this text chunk in arabic."
    )
    unique_10_arabic_keywords: str = Field(
        ..., description="10 unique keywords for this text chunk"
    )
    arabic_questions_this_excerpt_can_answer : List[str] = Field(
        ..., description="3 questions this context can provide specific answers to which are unlikely to be found elsewhere in arabic language, return nothing if it doesnt have context meaning"
    )

def extract_metadata(llm, docs, parser , presist_dir = "text_with_metadata/06-05-2016"):
    openai_program = OpenAIPydanticProgram.from_defaults(
        output_cls=NodeMetadata,
        prompt_template_str="{input}",
       # extract_template_str=EXTRACT_TEMPLATE_STR
    )

    program_extractor = PydanticProgramExtractor(
        program=openai_program, input_key="input", show_progress=True
    )
    extractors = [
        parser,
        #program_extractor,
        #EntityExtractor(prediction_threshold=0.5,device="cpu",label_entities=False)
        #TitleExtractor(nodes=5, llm=llm),
        SummaryExtractor(llm=llm,prompt_template=SUMMARY_PROMPT_TEMPLATE),
#        QuestionsAnsweredExtractor(questions=3, llm=llm,prompt_template=DEFAULT_QUESTION_GEN_TMPL, embedding_only=True),
        #KeywordExtractor(keywords=10, llm=llm),
        # CustomExtractor()
    ]
    pipeline = IngestionPipeline(transformations=extractors)
    nodes = pipeline.run(documents=docs) 
    os.mkdir(presist_dir)
    for node in nodes:
        with open(f"{presist_dir}/page_{node.metadata['page_number']}.json","w",encoding='utf-8') as file:
            json.dump(node.metadata, file, ensure_ascii=False, indent=4)
    return nodes

def get_vector_retreiver(nodes,document_name, top_k):
    data_store_path = "data_store"
    if not os.path.exists(f"{data_store_path}/{document_name}"):
        os.mkdir(f"{data_store_path}/{document_name}")
        print("start vector storing")
        vector_index = VectorStoreIndex(nodes=nodes,show_progress=True)
        vector_index.storage_context.persist(persist_dir=f"{data_store_path}/{document_name}")
    else:
        storage_context = StorageContext.from_defaults(persist_dir=f"{data_store_path}/{document_name}")
        vector_index = load_index_from_storage(storage_context)
    vector_retriever = VectorIndexRetriever(
        index=vector_index,
        similarity_top_k=top_k,
    )
    return vector_retriever


def get_graph_retirever(nodes,document_name,top_k):
    graph_store_path = "graph_store"
    graph_store = SimpleGraphStore()
    edge_types, rel_prop_names = ["relationship"], [
        "relationship"
    ]  # default, could be omit if create from an empty kg
    tags = ["entity"]  # default, could be omit if create from an empty kg
    if not os.path.exists(f"{graph_store_path}/{document_name}"):
        graph_storage_context = StorageContext.from_defaults(graph_store=graph_store)
        print("start graph storing")
        kg_index = KnowledgeGraphIndex(
            nodes=nodes,
            storage_context=graph_storage_context,
            max_triplets_per_chunk=10,
            edge_types=edge_types,
            rel_prop_names=rel_prop_names,
            tags=tags,
            include_embeddings=True,
            show_progress=True,
        )
        os.mkdir(f"{graph_store_path}/{document_name}")
        kg_index.storage_context.persist(persist_dir=f"{graph_store_path}/{document_name}")
        print("finished")
    else:
        graph_storage_context = StorageContext.from_defaults(persist_dir=f"{graph_store_path}/{document_name}",graph_store=graph_store)
        kg_index = load_index_from_storage(
            storage_context=graph_storage_context,
            max_triplets_per_chunk=10,
            edge_types=edge_types,
            rel_prop_names=rel_prop_names,
            tags=tags,
            include_embeddings=True,
        )
    kg_retriever = KGTableRetriever(
        index=kg_index, 
        retriever_mode="keyword",
        include_text=True, 
        similarity_top_k = top_k,
        num_chunks_per_query=5,
    )
    return kg_retriever

def get_bm25_retreiver(nodes,top_k):
    retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=top_k)
    return retriever



def get_combined_nodes(name,retrieved_nodes,query):
    combined_nodes = {}
    retrieved_nodes_ids = []
    for nodes in retrieved_nodes:
        node_ids = []
        for n in nodes:
            node_id = n.metadata.get("page_number",None)
            if node_id == None:
                continue
            combined_nodes[node_id] = n
            node_ids.append(node_id)
        retrieved_nodes_ids.append(node_ids)

    union_nodes = set(retrieved_nodes_ids[0])
    for node_ids in retrieved_nodes_ids[1:]:
        union_nodes = union_nodes.union(node_ids)
    reranker = SentenceTransformerRerank(top_n=3, model="BAAI/bge-reranker-base")
    reranked_nodes = reranker.postprocess_nodes(
        [combined_nodes[node_id] for node_id in union_nodes],
        query_bundle=QueryBundle(
            query
        ),
    )
    return {
        name : reranked_nodes,
    }    

def get_most_concurrent_nodes(retrievers_nodes):
        id_counts = {}
        nodes_with_ids = {}
        
        # Iterate over each engine's list of IDs
        for retreiver_nodes in retrievers_nodes:
            for key,nodes in retreiver_nodes.items():
                # Convert the list to a set to remove duplicates and then iterate
                for n in nodes:
                    node_id = n.metadata.get("page_number",None)
                    if not node_id:
                        continue
                    nodes_with_ids[node_id] = n
                    if node_id in id_counts:
                        id_counts[node_id] += 1
                    else:
                        id_counts[node_id] = 1
        total_counts = sum(id_counts.values())
        average_count = total_counts / len(id_counts)
        # Filter IDs that appear in two or more engine lists
        common_ids = [id for id, count in id_counts.items() if count >= len(retrievers_nodes)/2 ]

        return [nodes_with_ids[node_id] for node_id in common_ids]  

def basic_retirever(query):
    document_name = "06-20-2023"
    llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo", max_tokens=512)
    filename_fn = lambda file_path: {"date": file_path.split("/")[1],"page_number" : file_path.split("/")[-1].split(".")[0].split("_")[-1]}
    documents = SimpleDirectoryReader(f"./ocr_data/{document_name}",file_metadata=filename_fn).load_data()
    documents = sorted(documents,key=lambda doc : int(doc.metadata["page_number"]))
    parser = SentenceSplitter(chunk_size=2028)
    if not os.path.exists(f"text_with_metadata/{document_name}"):
        nodes = extract_metadata(llm=llm, docs=documents, parser=parser, presist_dir=f"text_with_metadata/{document_name}")
    else:
        nodes = parser.get_nodes_from_documents(documents)
        metadata_files = Path(f"text_with_metadata/{document_name}").glob("*.json")
        for i,file in enumerate(sorted(metadata_files, key=extract_number)):
            with open(file, 'r', encoding='utf-8') as json_file:
                metadata=json.load(json_file)
                nodes[i].metadata=metadata
    top_k = 5
    vector_retriever = get_vector_retreiver(nodes,document_name,top_k)
    nodes = vector_retriever._index.docstore.docs.values()
    bm25_retreiver = get_bm25_retreiver(nodes,top_k)
    kg_retreiver = get_graph_retirever(nodes, document_name,top_k)
    retrievers = {
        "vector_retriever" : vector_retriever,
        "kg_retreiver" : kg_retreiver,
        "bm25_retreiver" : bm25_retreiver,
    }
    retirieved_nodes = {}
    for key,retriever in retrievers.items():
        retirieved_nodes[key] = retriever.retrieve(query)
    kg_with_bm25_nodes = get_combined_nodes("kg_with_bm25_nodes",[retirieved_nodes["kg_retreiver"],retirieved_nodes["bm25_retreiver"]], query)
    vector_with_bm25_nodes = get_combined_nodes("vector_with_bm25_nodes",[retirieved_nodes["vector_retriever"],retirieved_nodes["bm25_retreiver"]], query)
    vector_with_kg_nodes = get_combined_nodes("vector_with_kg_nodes",[retirieved_nodes["vector_retriever"],retirieved_nodes["kg_retreiver"]], query)
    retirieved_nodes_by_retreivers = [kg_with_bm25_nodes,vector_with_bm25_nodes,vector_with_kg_nodes]
    most_occured_nodes = get_most_concurrent_nodes(retirieved_nodes_by_retreivers)
    return {
        **{"most_occured" : sorted(most_occured_nodes, key=lambda n: n.score, reverse=True)}
    }


