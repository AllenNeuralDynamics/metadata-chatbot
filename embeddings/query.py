# import boto3, json, os, pickle

# model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
# bedrock = boto3.client(
#     service_name="bedrock-runtime",
#     region_name = 'us-west-2'
# )

# from langchain_aws import BedrockEmbeddings
# embeddings_model = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=bedrock)

# result = (embeddings_model.embed_query("Which modality has the most experiments in the database?"))


# with open("query_vector", "wb") as fp:   #Pickling
#     pickle.dump(result, fp)

# with open("query_vector", "rb") as fp:   # Unpickling
#     b = pickle.load(fp)

# print(len(b))
import re, json
from langchain.docstore.document import Document 
from langchain_text_splitters import RecursiveJsonSplitter

splitter = RecursiveJsonSplitter(max_chunk_size=500)

with open("metadata_vector_index.data_assets.json") as f:
    json_doc = json.load(f)[0]

def regex_modality_PHYSIO(record_name: str) -> bool:

    PHYSIO_modalities = ["behavior", "Other", "FIP", "phys", "HSFP"]
    #SPIM_modalities = ["SPIM", "HCR"]

    PHYSIO_pattern = '(' + '|'.join(re.escape(word) for word in PHYSIO_modalities) + ')_'
    regex = re.compile(PHYSIO_pattern)

    return bool(regex.search(record_name))

def json_to_langchain_doc(json_doc: dict) -> list:

    docs = []

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
        if item in fields_to_embed:
            values_to_embed[item] = value
        else:
            to_metadata[item] = value
    json_chunks = splitter.split_text(json_data=values_to_embed)
    for chunk in json_chunks:
        newDoc = Document(page_content=chunk, metadata=to_metadata)
        docs.append(newDoc)

    return docs

print(json_to_langchain_doc(json_doc)[0])