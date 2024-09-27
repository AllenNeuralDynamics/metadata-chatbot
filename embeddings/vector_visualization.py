from langchain_community.vectorstores.documentdb import DocumentDBVectorSearch
from urllib.parse import quote_plus
import pymongo, os, boto3, re, pickle, json
from pymongo import MongoClient
from langchain_aws import BedrockEmbeddings
from sshtunnel import SSHTunnelForwarder
import logging

import numpy as np
from sklearn.manifold import TSNE
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio

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

logging.basicConfig(filename='vector_visualization.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filemode="w")


bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name = 'us-west-2'
)

BEDROCK_EMBEDDINGS = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock)

escaped_username = quote_plus(os.getenv("DOC_DB_USERNAME"))
escaped_password = quote_plus(os.getenv('DOC_DB_PASSWORD'))

CONNECTION_STRING = f"mongodb://{escaped_username}:{escaped_password}@localhost:27017/?directConnection=true&authMechanism=SCRAM-SHA-1&retryWrites=false"

INDEX_NAME = 'langchain_vector_embeddings_index'
NAMESPACE = 'metadata_vector_index.data_assets'
DB_NAME, COLLECTION_NAME = NAMESPACE.split(".")

client = MongoClient(CONNECTION_STRING)
collection = client[DB_NAME][COLLECTION_NAME]
langchain_collection = client[DB_NAME]['LANGCHAIN_data_assets']

LANGCHAIN_NAMESPACE = 'metadata_vector_index.LANGCHAIN_data_assets'

try:

    ssh_server = create_ssh_tunnel()
    ssh_server.start()
    logging.info("SSH tunnel opened")
    
    logging.info("Successfully connected to MongoDB")

    logging.info("Initializing connection vector store")

    vectorstore = DocumentDBVectorSearch.from_connection_string(
                    connection_string=CONNECTION_STRING,
                    namespace=LANGCHAIN_NAMESPACE,
                    embedding=BEDROCK_EMBEDDINGS,
                    index_name=INDEX_NAME
                )

    query = "What is the experimental history for mouse with subject id 719360"
    logging.info("Starting to vectorize query...")
    query_embedding = BEDROCK_EMBEDDINGS.embed_query(query)

    result = langchain_collection.aggregate([
        {
        '$search': {
            'vectorSearch': {
                'vector': query_embedding, 
                'path': 'vectorContent', 
                'similarity': 'cosine', 
                'k': 50
                }
            }
        }
    ])

    embeddings_list = []

    for document in result:
        embeddings_list.append(document['vectorContent'])

    embeddings_list.insert(0,query_embedding)

    print(len(embeddings_list))

    n_components = 3 #3D #TODO: PCA
    embeddings_list = np.array(embeddings_list) #converting to numpy array

    print(np.shape(embeddings_list))

    

    tsne = TSNE(n_components=n_components, random_state=42, perplexity=20)
    reduced_vectors = tsne.fit_transform(embeddings_list)
    print(len(reduced_vectors))
    #reduced_vectors[0:10]

        # Create a 3D scatter plot
    scatter_plot = go.Scatter3d(
        x=reduced_vectors[:, 0],
        y=reduced_vectors[:, 1],
        z=reduced_vectors[:, 2],
        mode='markers',
        marker=dict(size=5, color='grey', opacity=0.5, line=dict(color='lightgray', width=1)),
        text=[f"Point {i}" for i in range(len(reduced_vectors))]
    )

    # Highlight the first point with a different color
    highlighted_point = go.Scatter3d(
        x=[reduced_vectors[0, 0]],
        y=[reduced_vectors[0, 1]],
        z=[reduced_vectors[0, 2]],
        mode='markers',
        marker=dict(size=8, color='red', opacity=0.8, line=dict(color='lightgray', width=1)),
        text=["Question"]
        
    )

    blue_points = go.Scatter3d(
        x=reduced_vectors[1:4, 0],
        y=reduced_vectors[1:4, 1],
        z=reduced_vectors[1:4, 2],
        mode='markers',
        marker=dict(size=8, color='blue', opacity=0.8,  line=dict(color='black', width=1)),
        text=["Top 1 Data Asset","Top 2 Data Asset","Top 3 Data Asset"]
    )

    # Create the layout for the plot
    layout = go.Layout(
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z'),
        ),
        title=f'3D Representation after t-SNE (Perplexity=5)'
    )


    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scatter3d'}]])

    # Add the scatter plots to the Figure
    fig.add_trace(scatter_plot)
    fig.add_trace(highlighted_point)
    fig.add_trace(blue_points)

    fig.update_layout(layout)

    pio.write_html(fig, 'interactive_plot.html')
    fig.show()

    # try:
    #    vectorstore = DocumentDBVectorSearch.from_connection_string(
    #        connection_string=CONNECTION_STRING,
    #        namespace=NAMESPACE,
    #        embedding=embeddings_model,
    #        index_name=INDEX_NAME,
    #    )
    #    logging.info("Successfully connected to vector index")

    #    query = "test"
    #    logging.info("Starting to vectorize query...")
    #    query_embedding = embeddings_model.embed_query(query)
    # #    logging.debug(f"Query: {query}")
    # #    logging.debug(f"Query embedding (first 5 elements): {query_embedding[:5]}")
    #    docs = vectorstore.similarity_search(query)
    #    print(docs[0].page_content)
       
    # except Exception as e:
    #    logging.error(f"Error initializing DocumentDBVectorSearch: {e}")
    


except pymongo.errors.ServerSelectionTimeoutError as e:
    print(f"Server selection timeout error: {e}")
    print(f"Current topology description: {client.topology_description}")
except Exception as e:
    logging.exception(e)
finally:
    client.close()
    ssh_server.stop()
