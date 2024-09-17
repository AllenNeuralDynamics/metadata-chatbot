import boto3, os
from langchain_community.document_loaders.mongodb import MongodbLoader
from aind_data_access_api.document_db_ssh import DocumentDbSSHClient, DocumentDbSSHCredentials
from pymongo import MongoClient
from urllib.parse import quote_plus


#establishing embedding model
model_id = "amazon.titan-embed-text-v2:0"

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name = 'us-west-2'
)

print("hi")
from pymongo.errors import ConnectionFailure


connection_string = f"mongodb://{escaped_username}:{escaped_password}@localhost:27017/metadata_vector_index"

print(connection_string)
def test_connection(connection_string):
    try:
        print("connecting")
        client = MongoClient(connection_string, serverSelectionTimeoutMS=5000)
        client.admin.command('ismaster')
        print("MongoDB connection successful!")
        return True
    except ConnectionFailure:
        print("MongoDB connection failed!")
        return False
    
test_connection(connection_string)
       
# connecting to MongoDB

db = 'metadata_vector_index'
collection = 'data_assets_dev'

client = MongoClient(os.environ['CONNECTION_STRING'])
db = client['metadata_vector_index']
collection = db['data_assets_dev']

loader = MongodbLoader(
    connection_string = os.environ['CONNECTION_STRING'],
    db_name = 'metadata_vector_index',
    collection_name='data_assets_dev'
)

docs = loader.load()

len(docs)