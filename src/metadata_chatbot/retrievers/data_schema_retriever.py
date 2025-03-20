"""DocDB retriever class that communicates with MongoDB"""

import asyncio
import json
import logging
from typing import Any, List, Optional

from aind_data_access_api.document_db import MetadataDbClient
from bson import json_util
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field
from pymongo.collection import Collection
from sentence_transformers import SentenceTransformer

API_GATEWAY_HOST = "api.allenneuraldynamics-test.org"
DATABASE = "metadata_vector_index"
# COLLECTION = "aind_data_schema_vectors"


class DataSchemaRetriever(BaseRetriever):
    """
    A retriever that contains the top k documents, retrieved
    from the DocDB index, aligned with the user's query.
    """

    k: int = Field(default=5, description="Number of documents to retrieve")
    collection: str = Field(description="MongoDB collection to retrieve from")
    _model = None

    def _get_relevant_documents(
        self, query: str, **kwargs: Any
    ) -> List[Document]:
        """Synnchronous retriever"""

        return asyncio.run(self._aget_relevant_documents(query, **kwargs))

    async def _aget_relevant_documents(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Asynchronous retriever"""

        docdb_api_client = MetadataDbClient(
            host=API_GATEWAY_HOST,
            database=DATABASE,
            collection=self.collection,
        )

        # Embed query
        logging.info("connecting to embedding model")
        model = SentenceTransformer(
            "mixedbread-ai/mxbai-embed-large-v1", truncate_dim=1024
        )
        logging.info("Starting to embed query")
        embedded_query = model.encode(query, prompt_name="query")
        logging.info("Finished embed query")

        # Construct aggregation pipeline
        vector_search = {
            "$search": {
                "vectorSearch": {
                    "vector": embedded_query.tolist(),
                    "path": "vector_embeddings",
                    "similarity": "cosine",
                    "k": self.k,
                    "efSearch": 250,
                }
            }
        }

        projection_stage = {"$project": {"text": 1, "_id": 0}}

        pipeline = [vector_search, projection_stage]

        try:
            logging.info("Starting vector search")
            result = docdb_api_client.aggregate_docdb_records(
                pipeline=pipeline
            )

            # Process documents in a batch
            documents = []
            for document in result:
                json_doc = json.loads(json_util.dumps(document))
                page_content = json_doc.get("text", "")
                # Create metadata by excluding the text field
                metadata = {k: v for k, v in json_doc.items() if k != "text"}
                documents.append(
                    Document(page_content=page_content, metadata=metadata)
                )

            return documents

        except Exception as e:
            print(e)


# query = "How many injections were performed across all animals in the thalamus in the middle project using the following coordinate: AP: 2.8, ML: 0.2, DV: 0.6?"


# retriever = DataSchemaRetriever(k=2, collection = "mongodb_node_vectors")
# documents = retriever._get_relevant_documents(query=query)
# print(documents)
