import sys, os, json
from typing import List, Optional, Any, Union, Annotated
from pymongo.collection import Collection
from motor.motor_asyncio import AsyncIOMotorCollection
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from bson import json_util
from langsmith import trace as langsmith_trace
from pydantic import Field
from aind_data_access_api.document_db import MetadataDbClient

sys.path.append(os.path.abspath("C:/Users/sreya.kumar/Documents/GitHub/metadata-chatbot"))
from metadata_chatbot.utils import BEDROCK_EMBEDDINGS

API_GATEWAY_HOST = "api.allenneuraldynamics-test.org"
DATABASE = "metadata_vector_index"
COLLECTION = "bigger_LANGCHAIN_curated_chunks"

docdb_api_client = MetadataDbClient(
   host=API_GATEWAY_HOST,
   database=DATABASE,
   collection=COLLECTION,
)


class DocDBRetriever(BaseRetriever):
    """A retriever that contains the top k documents, retrieved from the DocDB index, aligned with the user's query."""
    #collection: Any = Field(description="DocDB collection to retrieve from")
    k: int = Field(default=10, description="Number of documents to retrieve")

    def _get_relevant_documents(
        self, 
        query: str, 
        query_filter: Optional[dict] = None,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs: Any,
    ) -> List[Document]:
        
        #Embed query
        embedded_query = BEDROCK_EMBEDDINGS.embed_query(query)

        #Construct aggregation pipeline
        vector_search = {
            "$search": { 
                "vectorSearch": { 
                    "vector": embedded_query, 
                    "path": 'vectorContent', 
                    "similarity": 'euclidean', 
                    "k": self.k
                }
            }
        }

        pipeline = [vector_search]
        if query_filter:
            pipeline.insert(0, query_filter)
    
        result = docdb_api_client.aggregate_docdb_records(pipeline=pipeline)

        page_content_field = 'textContent'

        results = []
        
        #Transform retrieved docs to langchain Documents
        for document in result:
            values_to_metadata = dict()

            json_doc = json.loads(json_util.dumps(document))

            for key, value in json_doc.items():
                if key == page_content_field:
                    page_content = value
                else:
                    values_to_metadata[key] = value

            new_doc = Document(page_content=page_content, metadata=values_to_metadata)
            results.append(new_doc)

        return results
    
    async def _aget_relevant_documents(
        self, 
        query: str, 
        query_filter: Optional[dict] = None,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs: Any,
    ) -> List[Document]:
        
        #Embed query
        embedded_query = BEDROCK_EMBEDDINGS.embed_query(query)

        #Construct aggregation pipeline
        vector_search = {
            "$search": { 
                "vectorSearch": { 
                    "vector": embedded_query, 
                    "path": 'vectorContent', 
                    "similarity": 'euclidean', 
                    "k": self.k,
                    "efSearch": 40
                }
            }
        }

        pipeline = [vector_search]
        if query_filter:
            pipeline.insert(0, query_filter)
    
        result = docdb_api_client.aggregate_docdb_records(pipeline=pipeline)
        #results = await cursor.to_list(length=1000)

        page_content_field = 'textContent'

        results = []
        
        #Transform retrieved docs to langchain Documents
        for document in result:
            values_to_metadata = dict()

            json_doc = json.loads(json_util.dumps(document))

            for key, value in json_doc.items():
                if key == page_content_field:
                    page_content = value
                else:
                    values_to_metadata[key] = value

            new_doc = Document(page_content=page_content, metadata=values_to_metadata)
            results.append(new_doc)

        return results