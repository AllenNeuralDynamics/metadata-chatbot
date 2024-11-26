from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

import logging, asyncio, uuid

from metadata_chatbot.agents.async_workflow import async_app
from metadata_chatbot.agents.workflow import app

from langchain_core.messages import AIMessage, HumanMessage
from streamlit.runtime.scriptrunner import add_script_run_ctx

from typing import Optional, List, Any, AsyncIterator
from langchain.callbacks.manager import AsyncCallbackManager, CallbackManagerForLLMRun
import streamlit as st



class GAMER(LLM):

    def _call(
        self,
        query: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Args:
            query: Natural language query.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            The model output as a string.
        """
        inputs = {"query" : query}
        answer = app.invoke(inputs)
        return answer['generation']
    
    async def _acall(
        self,
        query: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Asynchronous call.
        """
        async def main(query):
        
            unique_id =  str(uuid.uuid4())
            config = {"configurable":{"thread_id": unique_id}}
            inputs = {
                "messages": [HumanMessage(query)], 
            }
            async for output in async_app.astream(inputs, config):
                for key, value in output.items():
                    if key != "database_query":
                        yield value['messages'][0].content 
        
        curr = None
        generation = None
        async for result in main(query):
            if curr != None:
                print(curr)
            curr = generation
            generation = result
        return generation

    def _stream(
        self,
        query: str,
        unique_id: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream the LLM on the given prompt.
        """
        for char in query[: self.n]:
            chunk = GenerationChunk(text=char)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

            yield chunk

    async def streamlit_astream(
        self,
        query: str,
        unique_id: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Asynchronous call.
        """
        
        config = {"configurable":{"thread_id": unique_id}}
        inputs = {
            "messages": [HumanMessage(query)], 
        }
        async def main(query: str):
        
            unique_id =  str(uuid.uuid4())
            config = {"configurable":{"thread_id": unique_id}}
            inputs = {
                "messages": [HumanMessage(query)], 
            }
            async for output in async_app.astream(inputs, config):
                for key, value in output.items():
                    if key != "database_query":
                        yield value['messages'][0].content 
                    else:
                        yield value['generation']
        
        curr = None
        generation = None
        async for result in main(query):
            if curr != None:
                st.write(curr)
            curr = generation
            generation = result
        return generation
            


    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            "model_name": "Anthropic Claude 3 Sonnet",
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "Claude 3 Sonnet"
    
# llm = GAMER()

# async def main():
#     query = "Can you list all the procedures performed on the specimen, including their start and end dates? in SmartSPIM_662616_2023-03-06_17-47-13"
#     result = await llm.ainvoke(query)
#     print(result)

# asyncio.run(main())

# async def main():
#     result = await llm.ainvoke("Can you give me a timeline of events for subject 675387?")
#     print(result)

# asyncio.run(main())