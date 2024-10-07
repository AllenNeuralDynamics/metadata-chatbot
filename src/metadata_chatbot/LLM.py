from langchain_aws import ChatBedrock
from pydantic import BaseModel
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langsmith import trace as langsmith_trace
import logging, os, sys
from pymongo import MongoClient
from langchain.chains import RetrievalQA
from langchain import hub
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

from docdb_retriever import DocDBRetriever


sys.path.append(os.path.abspath("C:/Users/sreya.kumar/Documents/GitHub/metadata-chatbot"))
from utils import create_ssh_tunnel, CURATED_VECTORSTORE, CONNECTION_STRING, BEDROCK_CLIENT, ALL_CURATED_VECTORSTORE, LANGCHAIN_COLLECTION

logging.basicConfig(filename='LLM.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filemode="w")

MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

# class StructuredOutput(BaseModel):
#     '''An answer to the user question along with justification for the answer.'''
#     answer: str
#     reasoning: str

LLM = ChatBedrock(
    model_id= MODEL_ID,
    model_kwargs= {
        "temperature": 0
    }
)

metadata_field_info = [
    AttributeInfo(
        name="original_id",
        description="Serves as a reference to how the asset is labelled in the original database before chunking",
        type="string",
    ),
    AttributeInfo(
        name="created",
        description="Time stamp of when document was created",
        type="string",
    ),
    AttributeInfo(
        name="last_modified",
        description="When document was last modified",
        type="string",
    ),
    AttributeInfo(
        name="location", 
        description="Where data asset is stored in S3", 
        type="string"
    ),
    AttributeInfo(
        name="name", 
        description="File name, contains information about primary experimental modality used during the experiment, subject ID and when experiment was conducted", 
        type="string"
    ),
    AttributeInfo(
        name="processing", 
        description="How data was processed after uploading", 
        type= "dict"
    ),
    AttributeInfo(
        name="schema_version", 
        description="Schema version of current asset", 
        type="string"
    ),
    AttributeInfo(
        name="subject_id", 
        description="Identifier of subject experimented on", 
        type="string"
    ),
    AttributeInfo(
        name="modality", 
        description="Different experimental modalities used", 
        type="list"
    )
]

document_content_description = "Metadata acquired in lieu with experimental data acquistion"

try:

    ssh_server = create_ssh_tunnel()
    ssh_server.start()
    logging.info("SSH tunnel opened")

    client = MongoClient(CONNECTION_STRING)
    logging.info("Successfully connected to MongoDB")


    logging.info("Connecting to vectorstore")
    vectorstore = ALL_CURATED_VECTORSTORE

    logging.info("Initializing retriever...")
    retriever = DocDBRetriever(
        collection = LANGCHAIN_COLLECTION
    )

    filter = {'$match': {'modality': {'$in': ['ecephys', 'icephys', 'pophys']}}}
    print(retriever._get_relevant_documents(query = "injections", k = 3, query_filter = filter))

    # retriever = SelfQueryRetriever.from_llm(
    #     LLM, 
    #     vectorstore, 
    #     document_content_description, 
    #     metadata_field_info, 
    #     verbose=True
    #     )
    
    #print(retriever.get_relevant_documents("621025"))

    # prompt = hub.pull("eden19/modality_type")
    
    # qa_chain = RetrievalQA.from_chain_type(
    #     LLM, 
    #     retriever=retriever
    # )

    # logging.info("Constructing query...")
    # query = "621025 procedures"
    # logging.info("Query constructed")
    

    # result = qa_chain.invoke({"query": query, "prompt": prompt})
    # print(result)

except Exception as e:
    logging.exception(e)
finally:
    client.close()
    ssh_server.stop()

