import os
from typing import Sequence

from llama_index import VectorStoreIndex, StorageContext, load_index_from_storage, SimpleDirectoryReader
from llama_index.bridge.pydantic import BaseModel, Field
from ingestion import MetadataExtractor


class NawalicVectorIndex(BaseModel):
    ocr_data_dir : str = Field(description="Dir path for processed ocr data")
    document_metadata_dir : str = Field(description="Dir path for extracted metadata")
    vector_data_dir : str = Field(description="Directory path for vector store data")

    def _load_ocr_data(self):
        filename_fn = lambda file_path: {
            "page_number" : file_path.split("/")[-1].split(".")[0].split("_")[-1],
            "الفصل" : file_path.split("/")[1],
            "الدور" : file_path.split("/")[2],
            "المضبطة" : file_path.split("/")[3],
            "التاريخ": file_path.split("/")[4],
        }
        unstructred_documents = SimpleDirectoryReader(self.ocr_data_dir,file_metadata=filename_fn,recursive=True).load_data()
        documents = {}
        for doc in unstructred_documents:
            documents.setdefault(doc.metadata["التاريخ"],[])
            documents[doc.metadata["التاريخ"]].append(doc)
        return documents

    def _get_nodes_with_metadata(self):
        documents = self._load_ocr_data()
        nodes = []
        for key ,docs in documents.items():
            docs = sorted(docs,key=lambda doc : int(doc.metadata["page_number"]))
            node_loader = MetadataExtractor(documents=docs,document_key=key)
            nodes.extend(node_loader.get_nodes()) 
        return nodes
    
    def get_vector_index(self):
        if not os.path.exists(self.vector_data_dir):
            os.mkdir(self.vector_data_dir)
            nodes = self._get_nodes_with_metadata()
            print("start vector storing")
            vector_index = VectorStoreIndex(nodes=nodes,show_progress=True)
            vector_index.storage_context.persist(persist_dir=self.vector_data_dir)
        else:
            storage_context = StorageContext.from_defaults(persist_dir=self.vector_data_dir)
            vector_index = load_index_from_storage(storage_context)           
        return vector_index        
