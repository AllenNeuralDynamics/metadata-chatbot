import json, boto3, asyncio
from typing import List, Optional, Any
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from bson import json_util
from pydantic import Field
from aind_data_access_api.document_db import MetadataDbClient
from langchain_aws import BedrockEmbeddings

BEDROCK_CLIENT = boto3.client(
    service_name="bedrock-runtime",
    region_name = 'us-west-2'
)

BEDROCK_EMBEDDINGS = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=BEDROCK_CLIENT)

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
    k: int = Field(default=5, description="Number of documents to retrieve")

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
        embedded_query = await BEDROCK_EMBEDDINGS.aembed_query(query)

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
        
        #Transform retrieved docs to langchain Documents
        async def process_document(document):
            values_to_metadata = dict()
            json_doc = json.loads(json_util.dumps(document))

            for key, value in json_doc.items():
                if key == 'textContent':
                    page_content = value
                else:
                    values_to_metadata[key] = value
            return Document(page_content=page_content, metadata=values_to_metadata)
        
        tasks = [process_document(document) for document in result]
        result = await asyncio.gather(*tasks)

        return result