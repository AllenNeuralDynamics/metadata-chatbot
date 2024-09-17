from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

import boto3, json, os
from chat import get_completion
from botocore.exceptions import ClientError

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name = 'us-west-2'
)

#model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

class Metamorph(LLM):

    def _call(self, 
              prompt: str, 
              bedrock_client = bedrock, 
              stop: Optional[List[str]] = None, 
              run_manager: Optional[CallbackManagerForLLMRun] = None
              )-> str:
        answer = get_completion(prompt, bedrock_client)
        return answer
            
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {"model_name" : "Metamorph"}
    
    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "Claude 3 Sonnet"
    
if __name__ == '__main__': 
    llm = Metamorph()
    prompt = "Give me the count of genotypes in each modality in the database?"
    llm.invoke(prompt)
