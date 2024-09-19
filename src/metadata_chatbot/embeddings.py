from urllib.parse import quote_plus
import pymongo, os, boto3
from pymongo import MongoClient
from langchain_community.document_loaders.mongodb import MongodbLoader
from langchain_aws import BedrockEmbeddings
from sshtunnel import SSHTunnelForwarder
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from tqdm import tqdm

import logging

logging.basicConfig(filename='embeddings.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# Establishing bedrock client and embedding model 

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name = 'us-west-2'
)

embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock)

logging.info("Embedding model instantiated")

def _create_ssh_tunnel():
    """Create an SSH tunnel to the Document Database."""
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


username = os.getenv("DOC_DB_USERNAME")
password = os.getenv('DOC_DB_PASSWORD')
database_name = "metadata_vector_index"

   # Escape username and password to handle special characters
escaped_username = quote_plus(username)
escaped_password = quote_plus(password)

#connection_string = 'mongodb://localhost:27018/'
connection_string = f"mongodb://{escaped_username}:{escaped_password}@localhost:27017/?directConnection=true&authMechanism=SCRAM-SHA-1&retryWrites=false"

try:
    #print(f"Attempting to connect with: {connection_string}")

    ssh_server = _create_ssh_tunnel()
    ssh_server.start()
    logging.info("SSH tunnel opened")
    '''
    client = MongoClient(
            host="localhost",
            port=27017,
            retryWrites=False,
            directConnection=True,
            username=os.getenv('DOC_DB_USERNAME'),
            password=os.getenv('DOC_DB_PASSWORD'),
            authSource="admin",
            authMechanism="SCRAM-SHA-1",
        )
    '''

    client = MongoClient(connection_string)
    
    # Force a server check
    server_info = client.server_info()
    print(f"Server info: {server_info}")
    
    logging.info("Successfully connected to MongoDB")
    
    loader = MongodbLoader(
        connection_string = connection_string,
        db_name = 'metadata_vector_index',
        collection_name='data_assets'
    )

    logging.info("Loading collection...")

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=100
                    )

    #text_splitter = SemanticChunker(embeddings, breakpoint_threshold_type="gradient")
    docs = text_splitter.split_documents(documents)
    docs_text = [doc.page_content for doc in docs]

    logging.info("Embedding documents...")

    vectors = []

    for i in tqdm(range(len(docs_text)), desc="Embeddings in progress", total = len(docs_text)):
        vector = embeddings.embed_documents([docs_text[i]])[0]  # Embed a single document
        vectors.append(vector)
    logging.info("Embedding finished")
    print(len(vectors))

    print(len(docs))

except pymongo.errors.ServerSelectionTimeoutError as e:
    print(f"Server selection timeout error: {e}")
    print(f"Current topology description: {client.topology_description}")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    client.close()
    ssh_server.stop()
