"""Calls to Claude in Bedrock"""

from langchain_anthropic import ChatAnthropic
from langchain_aws.chat_models.bedrock import ChatBedrock

BEDROCK_SONNET_3_5 = "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
BEDROCK_HAIKU_3_5 = "us.anthropic.claude-3-5-haiku-20241022-v1:0"
BEDROCK_SONNET_3_7 = "us.anthropic.claude-3-7-sonnet-20250219-v1:0"

ANTHROPIC_SONNET_3_7 = "claude-3-7-sonnet-20250219"
ANTHROPIC_HAIKU_3_5 = "claude-3-5-haiku-latest"


SONNET_3_5_LLM = ChatBedrock(
    model=BEDROCK_SONNET_3_5,
    model_kwargs={"temperature": 0},
    streaming=True,
)

HAIKU_3_5_LLM = ChatBedrock(
    model=BEDROCK_HAIKU_3_5,
    model_kwargs={"temperature": 0},
    streaming=True,
)

SONNET_3_7_LLM = ChatBedrock(
    model=BEDROCK_SONNET_3_7,
    model_kwargs={"temperature": 0},
    streaming=True,
)

SONNET_PROMPT_CACHING = ChatAnthropic(model=ANTHROPIC_SONNET_3_7)

HAIKU_PROMPT_CACHING = ChatAnthropic(model=ANTHROPIC_HAIKU_3_5)

CLAUDE_REASONING = ChatAnthropic(
    model=ANTHROPIC_SONNET_3_7,
    max_tokens=5000,
    thinking={"type": "enabled", "budget_tokens": 2000},
)
