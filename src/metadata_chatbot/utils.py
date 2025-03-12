"""Calls to Claude in Bedrock"""

from langchain_aws.chat_models.bedrock import ChatBedrock

MODEL_ID_SONNET_3_5 = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
MODEL_ID_HAIKU_3_5 = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
MODEL_ID_SONNET_3_7 = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

SONNET_3_5_LLM = ChatBedrock(
    model_id=MODEL_ID_SONNET_3_5,
    model_kwargs={"temperature": 0},
    streaming=True,
)

HAIKU_3_5_LLM = ChatBedrock(
    model_id=MODEL_ID_HAIKU_3_5,
    model_kwargs={"temperature": 0},
    streaming=True,
)

SONNET_3_7_LLM = ChatBedrock(
    model_id=MODEL_ID_SONNET_3_7,
    model_kwargs={"temperature": 0},
    streaming=True,
)
