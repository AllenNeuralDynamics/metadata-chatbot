from sshtunnel import SSHTunnelForwarder
import logging, os, boto3
from urllib.parse import quote_plus
from langchain_community.vectorstores.documentdb import DocumentDBVectorSearch
from langchain_aws import BedrockEmbeddings
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient


BEDROCK_CLIENT = boto3.client(
    service_name="bedrock-runtime",
    region_name = 'us-west-2'
)

BEDROCK_EMBEDDINGS = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=BEDROCK_CLIENT)

escaped_username = quote_plus(os.getenv("DOC_DB_USERNAME"))
escaped_password = quote_plus(os.getenv('DOC_DB_PASSWORD'))

CONNECTION_STRING = f"mongodb://{escaped_username}:{escaped_password}@localhost:27017/?directConnection=true&authMechanism=SCRAM-SHA-1&retryWrites=false"

CURATED_NAMESPACE = 'metadata_vector_index.LANGCHAIN_curated_assets'
CURATED_INDEX_NAME = "curated_embeddings_index"

CURATED_VECTORSTORE = DocumentDBVectorSearch.from_connection_string(
            connection_string=CONNECTION_STRING,
            namespace=CURATED_NAMESPACE,
            embedding=BEDROCK_EMBEDDINGS,
            index_name=CURATED_INDEX_NAME
            )

client = MongoClient(CONNECTION_STRING)
LANGCHAIN_COLLECTION = client['metadata_vector_index']['LANGCHAIN_ALL_curated_assets']

ALL_curated_namespace = 'metadata_vector_index.LANGCHAIN_ALL_curated_assets'
ALL_curated_index_name = "ALL_curated_embeddings_index"
ALL_CURATED_VECTORSTORE = DocumentDBVectorSearch.from_connection_string(
            connection_string=CONNECTION_STRING,
            namespace=ALL_curated_namespace,
            embedding=BEDROCK_EMBEDDINGS,
            index_name=ALL_curated_index_name
            )


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

class ResourceManager:
    def __init__(self):
        self.ssh_server = None
        self.client = None
        self.async_client = None

    def __enter__(self):
        try:
            self.ssh_server = create_ssh_tunnel()
            self.ssh_server.start()
            logging.info("SSH tunnel opened")

            self.client = MongoClient(CONNECTION_STRING)
            self.async_client = AsyncIOMotorClient(CONNECTION_STRING)
            logging.info("Successfully connected to MongoDB")

            return self
        except Exception as e:
            logging.exception(e)
            self.__exit__(None, None, None)
            raise

    def __exit__(self, exc_type, exc_value, traceback):
        if self.client:
            self.client.close()
        if self.async_client:
            self.async_client.close()
        if self.ssh_server:
            self.ssh_server.stop()
        logging.info("Resources cleaned up")