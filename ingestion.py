import os
import json
import re

from typing import Sequence,Optional
from pathlib import Path
from llama_index.schema import BaseNode, Document
from llama_index import SimpleDirectoryReader, StorageContext
from llama_index.node_parser import SentenceSplitter
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.bridge.pydantic import BaseModel, Field
from llama_index.program.openai_program import OpenAIPydanticProgram
from llama_index.extractors import PydanticProgramExtractor
from llama_index.ingestion import IngestionPipeline
from llama_index.llms import OpenAI
from llama_index.extractors import SummaryExtractor,EntityExtractor

os.environ["OPENAI_API_KEY"] = "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"

def extract_number(filename):
    # Extract the number from the filename and return it as an integer
    match = re.search(r'\d+', filename.stem)
    return int(match.group()) if match else 0

SUMMARY_PROMPT_TEMPLATE = """\
Here is the content of the section:
{context_str}

Summarize the key topics and entities of the section. \

The summary must be in arabic
Summary: """



class PoliticMetadata(BaseModel):
    """
    Use for General purpose because it extract multiple metadata from one llm call
    """

    unique_5_keywords: str = Field(
        ..., description="5 unique keywords for this text chunk"
    )
    question_this_excerpt_can_answer : str = Field(
        ..., description="question this context can provide specific answers to which are unlikely to be found elsewhere in arabic language, return nothing if it doesnt have context meaning"
    )


class MetadataExtractor(BaseModel):
    """
    Metadata extractor to extract extra metadata from each chunk's content
    """
    documents: Sequence[Document] = Field(description="Documents to ingest")
    document_key: str = Field(description="Document full dir path")
    document_metadata_dir : str = Field(description="path for document's metadata store")
    metadata_llm_exclude : Optional[Sequence[str]] = Field(description="metadata that we want to exlude from the llmp prompt")

    def __init__(
        self,
        documents : Sequence[Document],
        document_key : str,
        document_metadata_dir : str = "metadata",
        metadata_llm_exclude : Sequence[str] = None,
        **kwargs,
    ):
        super().__init__(
            documents = documents,
            document_key = document_key,
            document_metadata_dir = document_metadata_dir,
            metadata_llm_exclude = metadata_llm_exclude,
            **kwargs,
        )

    def _save_metadata(self, nodes) -> None:
        """
        Save the metadata in local dir
        """
        metadata_dir = f"{self.document_metadata_dir}/{self.document_key}"
        os.makedirs(metadata_dir,exist_ok=True)
        for node in nodes:
            filename = f"{metadata_dir}/page_{node.metadata['page_number']}.json"
            data = [node.metadata,]
            if os.path.exists(filename):
                with open(filename, "r", encoding='utf-8') as file:
                    existing_data = json.load(file)
                data.extend(existing_data)
            with open(filename,"w+",encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
        return
    

    def get_nodes(self) -> Sequence[BaseNode]:
        """
        Split documents into chunks and extract the metadata
        """
        parser = SentenceSplitter(chunk_size=1024,chunk_overlap=105,separator="/n")
        openai_program = OpenAIPydanticProgram.from_defaults(
            output_cls=PoliticMetadata,
            prompt_template_str="{input}",
            extract_template_str=SUMMARY_PROMPT_TEMPLATE
        )
        program_extractor = PydanticProgramExtractor(
            program=openai_program, input_key="input", show_progress=True
        )
        llm = OpenAI(temperature=0, model="gpt-3.5-turbo", max_tokens=256)
        extractors = [
            parser,
            #program_extractor,
            #EntityExtractor(prediction_threshold=0.5,device="cpu",label_entities=False),
            #TitleExtractor(nodes=5, llm=llm),
            SummaryExtractor(llm=llm,prompt_template=SUMMARY_PROMPT_TEMPLATE),
            #QuestionsAnsweredExtractor(questions=3, llm=llm,prompt_template=DEFAULT_QUESTION_GEN_TMPL, embedding_only=True),
            #KeywordExtractor(keywords=10, llm=llm),
        ]
        pipeline = IngestionPipeline(transformations=extractors)
        print(f"Start metadata extraction for {self.document_key}")
        nodes = pipeline.run(documents=self.documents) 
        self._save_metadata(nodes)
        return nodes
        


