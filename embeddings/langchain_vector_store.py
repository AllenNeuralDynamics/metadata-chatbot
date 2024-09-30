from langchain_community.vectorstores.documentdb import (
    DocumentDBSimilarityType,
    DocumentDBVectorSearch,
)
from langchain.docstore.document import Document 
from urllib.parse import quote_plus
import pymongo, os, boto3, re, json
from pymongo import MongoClient
from langchain_aws import BedrockEmbeddings
from sshtunnel import SSHTunnelForwarder
import logging
from tqdm import tqdm

from bson import json_util

from langchain_text_splitters import RecursiveJsonSplitter

JSON_SPLITTER = RecursiveJsonSplitter(max_chunk_size=300)
TOKEN_LIMIT = 8192

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name = 'us-west-2'
)

BEDROCK_EMBEDDINGS = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock)

def create_ssh_tunnel():
    """Create an SSH tunnel to the Document Database."""
    try:
        return SSHTunnelForwarder(
            ssh_address_or_host=(
                os.getenv("DOC_DB_SSH_HOST"),
                22,
            ),
            ssh_username=os.getenv("DOC_DB_SSH_USERNAME"),
            ssh_password=os.getenv("DOC_DB_SSH_PASSWORD"),
            remote_bind_address=(os.getenv("DOC_DB_HOST"), 27017),
            local_bind_address=(
                "localhost",
                27017,
            ),
        )
    except Exception as e:
        logging.error(f"Error creating SSH tunnel: {e}")

def regex_modality_PHYSIO(record_name: str) -> bool:

    PHYSIO_modalities = ["behavior", "Other", "FIP", "phys", "HSFP"]
    #SPIM_modalities = ["SPIM", "HCR"]

    PHYSIO_pattern = '(' + '|'.join(re.escape(word) for word in PHYSIO_modalities) + ')_'
    regex = re.compile(PHYSIO_pattern)

    return bool(regex.search(record_name))


def json_to_langchain_doc(json_doc: dict) -> list:

    docs = []
    large_docs = []

    PHYSIO_fields_to_embed = ["rig", "session"]

    SPIM_fields_To_embed = ["instrument", "acquisition"]

    general_fields_to_embed = ["data_description", "subject", "procedures"]

    if regex_modality_PHYSIO(json_doc["name"]):
        fields_to_embed = [*PHYSIO_fields_to_embed, *general_fields_to_embed]
    else: 
        fields_to_embed = [*SPIM_fields_To_embed, *general_fields_to_embed]

    #fields_to_metadata = ["_id", "created", "describedBy", "external_links", "last_modified", "location", "metadata_status", "name", "processing", "schema_version"]

    to_metadata = dict()
    values_to_embed = dict()

    for item, value in json_doc.items():
        if item == "_id":
            item = "original_id"
        if item in fields_to_embed:
            values_to_embed[item] = value
        else:
            to_metadata[item] = value
    subject = json_doc.get("subject")
    
    if subject is not None:
        to_metadata["subject_id"] = subject.get("subject_id", "null")  # Default if subject_id is missing
    else:
        print("Subject key is missing or None.")
        to_metadata["subject_id"] = "null"
    json_chunks = JSON_SPLITTER.split_text(json_data=values_to_embed)


    for chunk in json_chunks:
        newDoc = Document(page_content=chunk, metadata=to_metadata)
        if len(chunk)//4 < TOKEN_LIMIT:
            docs.append(newDoc)
        else:
            large_docs.append(newDoc)
        break

        

    return docs, large_docs

logging.basicConfig(filename='vector_store.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filemode="w")

escaped_username = quote_plus(os.getenv("DOC_DB_USERNAME"))
escaped_password = quote_plus(os.getenv('DOC_DB_PASSWORD'))

CONNECTION_STRING = f"mongodb://{escaped_username}:{escaped_password}@localhost:27017/?directConnection=true&authMechanism=SCRAM-SHA-1&retryWrites=false"

INDEX_NAME = 'langchain_curated_embeddings_index'
NAMESPACE = 'metadata_vector_index.curated_assets'
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")

client = MongoClient(CONNECTION_STRING)
collection = client[DB_NAME][COLLECTION_NAME]
langchain_collection = client[DB_NAME]['LANGCHAIN_curated_assets']

LANGCHAIN_NAMESPACE = 'metadata_vector_index.LANGCHAIN_curated_assets'


try:

    ssh_server = create_ssh_tunnel()
    ssh_server.start()
    logging.info("SSH tunnel opened")
    
    logging.info("Successfully connected to MongoDB")

    logging.info(f"Finding assets that are already embedded...")

    if langchain_collection is not None:
        existing_ids = set(doc['original_id'] for doc in langchain_collection.find({}, {'original_id': 1}))

    logging.info(f"Skipped {len(existing_ids)} assets, which are already in the new collection")

    docs_to_vectorize = collection.count_documents({'_id': {'$nin': list(existing_ids)}})

    logging.info(f"{docs_to_vectorize} assets need to be vectorized")

    if docs_to_vectorize != 0:

        cursor = collection.find({'_id': {'$nin': list(existing_ids)}})

        docs = []
        skipped_docs = []

        logging.info("Chunking documents...")

        document_no = 0

        for document in tqdm(cursor, desc="Chunking in progress"):
            if document_no % 100 == 0:
                logging.info(f"Currently on asset number {document_no}")
            json_doc = json.loads(json_util.dumps(document))
            chunked_docs, large_docs = json_to_langchain_doc(json_doc)
            docs.extend(chunked_docs)
            skipped_docs.extend(large_docs)
            document_no += 1

        logging.info(f"Successfully chunked {document_no} documents")

        logging.info(f"Adding {len(docs)} chunked documents to collection")
        logging.info(f"Skipping {len(skipped_docs)} due to token limitations")

        for index, doc in enumerate(docs):
            try:
                vectorstore = DocumentDBVectorSearch.from_documents(
                    documents=docs,
                    embedding=BEDROCK_EMBEDDINGS,
                    collection=langchain_collection,
                    index_name=INDEX_NAME,
                )
            except Exception as e:
                logging.error(f"Error processing item at index {index}: {str(e)}")
                continue
            
        dimensions = 1024
        similarity_algorithm = DocumentDBSimilarityType.COS

        logging.info("Creating vector index with chunked documents")

        vectorstore.create_index(dimensions, similarity_algorithm)

    else:
        logging.info("Initializing connection to vector store")
        vectorstore = DocumentDBVectorSearch.from_connection_string(
                    connection_string=CONNECTION_STRING,
                    namespace=LANGCHAIN_NAMESPACE,
                    embedding=BEDROCK_EMBEDDINGS,
                    index_name=INDEX_NAME
                )
        
        query = "What is the experimental history for mouse with subject id 719360"

        docs = vectorstore.similarity_search(query)

        print(docs)

except pymongo.errors.ServerSelectionTimeoutError as e:
    print(f"Server selection timeout error: {e}")
    print(f"Current topology description: {client.topology_description}")
except Exception as e:
    logging.exception(e)
finally:
    client.close()
    ssh_server.stop()
