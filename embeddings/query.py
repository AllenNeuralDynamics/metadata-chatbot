import boto3, json, os, pickle

model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name = 'us-west-2'
)

from langchain_aws import BedrockEmbeddings
embeddings_model = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock)

result = (embeddings_model.embed_query("Which modality has the most experiments in the database?"))


with open("query_vector", "wb") as fp:   #Pickling
    pickle.dump(result, fp)

with open("query_vector", "rb") as fp:   # Unpickling
    b = pickle.load(fp)

print(len(b))