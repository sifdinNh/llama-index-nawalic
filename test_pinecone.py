import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, Document
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from llama_index.core.node_parser import SentenceSplitter
from pathlib import Path

os.environ["OPENAI_API_KEY"] = "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"

pc = Pinecone(
    api_key="3ba5b6e0-68c6-4dc6-be70-fe2038c02663"
)
#documents = SimpleDirectoryReader("ocr_data/12-22-2015").load_data()
documents = []
for page in Path("ocr_data/12-22-2015").glob("*.txt"):
    with open(page,"r") as file:
        text = file.read()
    page_number = page.name.split("_")[-1].replace(".txt","")
    doc = Document(text=text,metadata={"page":page_number,},doc_id="23dfs4")
    documents.append(doc)
parser = SentenceSplitter(chunk_size=512,chunk_overlap=60,separator="\n")
nodes = parser.get_nodes_from_documents(documents)
vector_store = PineconeVectorStore(pinecone_index=pc.Index("naw"))
#storage_context = StorageContext.from_defaults(vector_store=vector_store)
index = VectorStoreIndex.from_vector_store(vector_store=vector_store)
#index.insert_nodes(nodes)
retriever = index.as_retriever()
nodes = retriever.retrieve("استخرج الفهرس من هادا التاريخ 12-22-2015")
breakpoint()
print(nodes)