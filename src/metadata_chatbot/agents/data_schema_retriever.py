import json, asyncio
import numpy as np
from typing import List, Optional, Any
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from bson import json_util
from pydantic import Field
from aind_data_access_api.document_db import MetadataDbClient

from sentence_transformers import SentenceTransformer


API_GATEWAY_HOST = "api.allenneuraldynamics-test.org"
DATABASE = "metadata_vector_index"
COLLECTION = "aind_data_schema_vectors"

docdb_api_client = MetadataDbClient(
   host=API_GATEWAY_HOST,
   database=DATABASE,
   collection=COLLECTION,
)

class DataSchemaRetriever(BaseRetriever):
    """A retriever that contains the top k documents, retrieved from the DocDB index, aligned with the user's query."""
    k: int = Field(default=5, description="Number of documents to retrieve")

    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        '''Synnchronous retriever'''
        # This is a synchronous version of your retrieval method
        # You can either implement the logic here directly,
        # or use asyncio to run the async version
        return asyncio.run(self._aget_relevant_documents(query, **kwargs))
    
    async def _aget_relevant_documents(
        self, 
        query: str, 
        query_filter: Optional[dict] = None,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs: Any,
    ) -> List[Document]:
        '''Asynchronous retriever'''
        
        #Embed query
        query_to_embed = [query]
        query_prompt_name = "s2p_query"
        model = SentenceTransformer("dunzhang/stella_en_1.5B_v5", trust_remote_code=True)
        embedded_query = model.encode(query_to_embed, prompt_name=query_prompt_name)[0]
 
        #Construct aggregation pipeline
        vector_search = {
            "$search": { 
                "vectorSearch": { 
                    "vector": embedded_query.tolist(), 
                    "path": 'vector_embeddings', 
                    "similarity": 'cosine', 
                    "k": self.k,
                    "efSearch": 250
                }
            }
        }

        projection_stage = {
            "$project": {
                "text": 1,  
                "_id": 0  
            }
        }

        pipeline = [vector_search, projection_stage]
        if query_filter:
            pipeline.insert(0, query_filter)



        try:
            result = docdb_api_client.aggregate_docdb_records(pipeline=pipeline)
        except Exception as e:
            print(e)
        
        async def process_document(document):
            '''Asynchronously transforms retrieved docs to langchain documents'''
            values_to_metadata = dict()
            json_doc = json.loads(json_util.dumps(document))

            for key, value in json_doc.items():
                if key == 'text':
                    page_content = value
                else:
                    values_to_metadata[key] = value
            return Document(page_content=page_content, metadata=values_to_metadata)
        
        tasks = [process_document(document) for document in result]
        result = await asyncio.gather(*tasks)

        return result
    