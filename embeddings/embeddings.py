from urllib.parse import quote_plus
import pymongo, os, boto3, re, pickle
from pymongo import MongoClient
from langchain_community.document_loaders.mongodb import MongodbLoader
from langchain_aws import BedrockEmbeddings
from sshtunnel import SSHTunnelForwarder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from tqdm import tqdm
from transformers import AutoTokenizer
from datetime import datetime
from tqdm.contrib.logging import logging_redirect_tqdm

import logging

logging.basicConfig(filename='embeddings.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Constants
TOKEN_LIMIT = 8192 # TODO: update the value
BATCH_SIZE = 100


# Establishing bedrock client and embedding model 
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name = 'us-west-2'
)

embeddings_model = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock)

logging.info("Embedding model instantiated")

#Establishing tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

logging.info("Tokenizer instantiated")


#TODO : CONVERT FUNCTION TO normal tokenizer
def count_tokens(id, text):
    tokens = tokenizer.encode(text, truncation=False)
    print(f"{id}:{len(tokens)} tokens")
    if len(tokens) > TOKEN_LIMIT:
        logging.info(f"{id} has {len(tokens)} tokens.  Too large.")
        return None
    return tokens

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


def generate_embeddings_for_batch(client: MongoClient, batch: list) -> dict:
    """Generates embeddings vectors for a batch of loaded documents
    """
    logging.info("Embedding documents...")
    db = client["metadata_vector_index"]
    result_collection = db["data_assets_vectors"]
    
    skipped_ids = []
    failed_ids = []
    batch_vectors = dict()
    with logging_redirect_tqdm():
        for doc in tqdm(batch, desc="Embeddings in progress", total = len(batch)):
            
            doc_text = doc.page_content

            pattern = r"'_id':\s*'([^']+)'"
            match = re.search(pattern, doc_text)
            if match:
                id_value = match.group(1)
            else:
                # TODO: log warning
                continue

            if result_collection.count_documents({"_id": id_value, "vector_embeddings": {"$exists": True}}):
                skipped_ids.append(id_value)
                continue

            tokens = count_tokens(id_value, doc_text)
            if tokens is None:
                failed_ids.append(id_value)
                continue
            vector = embeddings_model.embed_documents([doc_text])[0]  # Embed a single document

            batch_vectors[id_value] = vector
    logging.info("Embedding finished for batch")
    logging.info(f"Succesfully embedded {len(batch_vectors)}/{len(batch)} documents.")
    logging.warning(f"Failed for {len(failed_ids)} documents: {failed_ids}")
    logging.info(f"Skipped {len(skipped_ids)} documents: {skipped_ids}")
    return batch_vectors

def write_embeddings_to_docdb_for_batch(client: MongoClient, batch_vectors: dict) -> None:
    db = client["metadata_vector_index"]
    result_collection = db["data_assets_vectors"]
    
    for id, vector in batch_vectors.items():
       logging.info(f"Adding vector embeddings for {id} to docdb")
       filter={"_id": id}
       update={
            "$set": {
                "vector_embeddings": vector
            }
        }
       result = result_collection.update_one(filter, update, upsert=False)
       logging.info(result.raw_result)
    return

database_name = "metadata_vector_index"

   # Escape username and password to handle special characters
escaped_username = quote_plus(os.getenv("DOC_DB_USERNAME"))
escaped_password = quote_plus(os.getenv('DOC_DB_PASSWORD'))

connection_string = f"mongodb://{escaped_username}:{escaped_password}@localhost:27017/?directConnection=true&authMechanism=SCRAM-SHA-1&retryWrites=false"

try:
    #print(f"Attempting to connect with: {connection_string}")

    ssh_server = create_ssh_tunnel()
    ssh_server.start()
    logging.info("SSH tunnel opened")

    client = MongoClient(connection_string)
    
    # Force a server check
    server_info = client.server_info()
    print(f"Server info: {server_info}")
    
    logging.info("Successfully connected to MongoDB")

    
    #possibly filter criteria subject/data desc etc
    loader = MongodbLoader(
        connection_string = connection_string,
        db_name = 'metadata_vector_index',
        collection_name='data_assets'
    )

    logging.info("Loading collection...")

    documents = loader.load()
    total_docs = len(documents)
    logging.info(f"Loaded {total_docs} documents..")

    #text_splitter = RecursiveCharacterTextSplitter(
    #                    chunk_size=1000,
    #                    chunk_overlap=100
    #                )

    #text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="gradient")
    #docs = text_splitter.split_documents(documents)
    #docs_text = [doc.page_content for doc in docs]


    for i in range(0, total_docs, BATCH_SIZE):
        end = i+BATCH_SIZE if i+BATCH_SIZE<total_docs else total_docs
        batch = documents[i:end]

        batch_vectors = generate_embeddings_for_batch(client=client, batch=batch)
        write_embeddings_to_docdb_for_batch(client=client, batch_vectors=batch_vectors)
        datestamp=datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f'vector_dictionary_{i}_{datestamp}.pkl', 'wb') as f:
            pickle.dump(batch_vectors, f)
        logging.info(f"Processed batch {i}")

    logging.info("Dictionary saved successfully.")

except pymongo.errors.ServerSelectionTimeoutError as e:
    print(f"Server selection timeout error: {e}")
    print(f"Current topology description: {client.topology_description}")
except Exception as e:
    logging.exception(e)
finally:
    client.close()
    ssh_server.stop()
