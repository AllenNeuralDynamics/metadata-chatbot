import boto3
from langchain_aws import BedrockEmbeddings
from langchain_aws.chat_models.bedrock import ChatBedrock

BEDROCK_CLIENT = boto3.client(
    service_name="bedrock-runtime",
    region_name = 'us-west-2'
)

BEDROCK_EMBEDDINGS = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=BEDROCK_CLIENT)

MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"

LLM = ChatBedrock(
    model_id= MODEL_ID,
    model_kwargs= {
        "temperature": 0
    }
)