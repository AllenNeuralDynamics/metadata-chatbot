from urllib.parse import quote_plus
import pymongo
from pymongo import MongoClient
from langchain_community.document_loaders.mongodb import MongodbLoader

username = "sreya.kumar"
password = "Nimalja3min!23"
database_name = "metadata_vector_index"

   # Escape username and password to handle special characters
escaped_username = quote_plus(username)
escaped_password = quote_plus(password)

connection_string = 'mongodb://localhost:27018/'

try:
    print(f"Attempting to connect with: {connection_string}")
    client = MongoClient('mongodb://localhost:27018/', serverSelectionTimeoutMS=5000)
    print("Initial connection successful")
    
    # Force a server check
    server_info = client.server_info()
    print(f"Server info: {server_info}")
    
    print("Connected successfully!")

except pymongo.errors.ServerSelectionTimeoutError as e:
    print(f"Server selection timeout error: {e}")
    print(f"Current topology description: {client.topology_description}")
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    if 'client' in locals():
        client.close()