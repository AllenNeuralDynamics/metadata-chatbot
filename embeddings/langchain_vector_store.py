from langchain_community.vectorstores.documentdb import (
    DocumentDBSimilarityType,
    DocumentDBVectorSearch,
)
from langchain.docstore.document import Document 
import pymongo, os, boto3, re, json, sys
from pymongo import MongoClient
from langchain_aws import BedrockEmbeddings
import logging
from tqdm import tqdm
from bson import json_util
from langchain_text_splitters import RecursiveJsonSplitter

sys.path.append(os.path.abspath("C:/Users/sreya.kumar/Documents/GitHub/metadata-chatbot"))
from metadata_chatbot.utils import create_ssh_tunnel, CONNECTION_STRING, BEDROCK_CLIENT, ResourceManager

logging.basicConfig(filename='vector_store.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filemode="w")

TOKEN_LIMIT = 8192
JSON_SPLITTER = RecursiveJsonSplitter(max_chunk_size=TOKEN_LIMIT)

BEDROCK_EMBEDDINGS = BedrockEmbeddings(model_id = "amazon.titan-embed-text-v2:0",client = BEDROCK_CLIENT)


def regex_modality_PHYSIO(record_name: str) -> bool:

    PHYSIO_modalities = ["behavior", "Other", "FIP", "phys", "HSFP"]
    #SPIM_modalities = ["SPIM", "HCR"]

    PHYSIO_pattern = '(' + '|'.join(re.escape(word) for word in PHYSIO_modalities) + ')_'
    regex = re.compile(PHYSIO_pattern)

    return bool(regex.search(record_name))


def json_to_langchain_doc(json_doc: dict) -> tuple[list, list]:

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
        to_metadata["subject_id"] = subject.get("subject_id", None)  # Default if subject_id is missing
    else:
        #print("Subject key is missing or None.")
        to_metadata["subject_id"] = "null"

    data_description = json_doc.get("data_description")

    if data_description is not None:
        to_metadata["modality"] = data_description.get("modality", None)  # Default if subject_id is missing
    else:
        #print("Subject key is missing or None.")
        to_metadata["modality"] = "null"

    json_chunks = JSON_SPLITTER.split_text(json_data=values_to_embed, convert_lists=True)

    for chunk in json_chunks:
        newDoc = Document(page_content=chunk, metadata=to_metadata)
        if len(chunk) < TOKEN_LIMIT:
            docs.append(newDoc)
        else:
            large_docs.append(newDoc)

    return docs, large_docs

#INDEX_NAME = 'ALL_curated_embeddings_index'
INDEX_NAME = 'TOKEN_LIMIT_curated_embeddings_index'
NAMESPACE = 'metadata_vector_index.curated_assets'
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")


with ResourceManager() as RM:

    collection = RM.client[DB_NAME][COLLECTION_NAME]
    langchain_collection = RM.client[DB_NAME]['bigger_LANGCHAIN_curated_chunks']
    LANGCHAIN_NAMESPACE = 'metadata_vector_index.bigger_LANGCHAIN_curated_chunks'

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

        try:
            vectorstore = DocumentDBVectorSearch(
                embedding=BEDROCK_EMBEDDINGS,
                collection=langchain_collection,
                index_name=INDEX_NAME,
            )

            batch_size = 100
            for i in range(0, len(docs), batch_size):
                batch = docs[i:i + batch_size]
                vectorstore.add_documents(batch)
                logging.info(f"Added batch {i // batch_size + 1} of documents")

            dimensions = 1024
            similarity_algorithm = DocumentDBSimilarityType.COS

            logging.info("Creating vector index with chunked documents")
            vectorstore.create_index(dimensions, similarity_algorithm)

        except Exception as e:
            logging.error(f"Error processing documents: {str(e)}")          

    else:
        logging.info("Vectorstore is up to date!")
        
