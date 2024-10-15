from langchain_community.vectorstores.documentdb import DocumentDBVectorSearch
from urllib.parse import quote_plus
import pymongo, os, boto3, sys, umap
from pymongo import MongoClient
from langchain_aws import BedrockEmbeddings
import logging
import matplotlib.pyplot as plt
import umap.umap_ as umap


import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import plotly.io as pio

sys.path.append(os.path.abspath("C:/Users/sreya.kumar/Documents/GitHub/metadata-chatbot"))
from metadata_chatbot.utils import create_ssh_tunnel, ALL_CURATED_VECTORSTORE, BEDROCK_EMBEDDINGS, CONNECTION_STRING


logging.basicConfig(filename='vector_visualization.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filemode="w")

client = MongoClient(CONNECTION_STRING)
langchain_collection = client['metadata_vector_index']['LANGCHAIN_ALL_curated_assets']



ssh_server = create_ssh_tunnel()
ssh_server.start()
logging.info("SSH tunnel opened")

logging.info("Successfully connected to MongoDB")
logging.info("Initializing connection vector store")
vectorstore = ALL_CURATED_VECTORSTORE

    # query = "subject"
    # logging.info("Starting to vectorize query...")
    # query_embedding = BEDROCK_EMBEDDINGS.embed_query(query)

    # total_documents = langchain_collection.count_documents({})
    # print(f"Total documents in collection: {total_documents}")

    # # Check indexes on the collection
    # indexes = langchain_collection.index_information()
    # print(f"Indexes on collection: {indexes}")

    # result = langchain_collection.aggregate([
    #     {
    #     '$search': {
    #         'vectorSearch': {
    #             'vector': query_embedding, 
    #             'path': 'vectorContent', 
    #             'similarity': 'cosine', 
    #             'k': 22100
    #             }
    #         }
    #     },
    #     {
    #         '$limit': 22100  # Ensure the pipeline limits the results to 22100
    #     }
    # ])

logging.info("Finding vectors...")
documents = langchain_collection.find({}, {"vectorContent": 1, "_id": 0})
logging.info("Extracting vectors...") \

embeddings_list = []
for doc in documents:
    embeddings_list.append(doc["vectorContent"])
logging.info(f"Number of vectors retrieved: {len(embeddings_list)}")
embeddings_array = np.array(embeddings_list)
logging.info("Plotting...")
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
embedding = reducer.fit_transform(embeddings_array)
# Plot the results
plt.figure(figsize=(12, 10))
plt.scatter(embedding[:, 0], embedding[:, 1], s=3, alpha=0.5)
plt.title(f'UMAP projection of {len(embeddings_list)} embeddings')
plt.xlabel('UMAP1')
plt.ylabel('UMAP2')
plt.colorbar()
plt.show()


'''
    embeddings_list = []
    modalities_list = []

    for document in result:
        embeddings_list.append(document['vectorContent'])
        modalities_list.append(document['modality'])

    #embeddings_list.insert(0,query_embedding)

    print(len(embeddings_list))
    print(len(modalities_list))

    n_components = 3 #3D 
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
        z1=reduced_vectors[1:4, 3],
        z2=reduced_vectors[1:4, 4],
        z3=reduced_vectors[1:4, 5],
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
'''

client.close()
ssh_server.stop()
