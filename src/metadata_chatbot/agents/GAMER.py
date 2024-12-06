from typing import Any, Dict, Iterator, List, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

import logging, asyncio, uuid

from metadata_chatbot.agents.async_workflow import async_app
from metadata_chatbot.agents.workflow import app
from async_workflow import async_app

from langchain_core.messages import AIMessage, HumanMessage

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
        async def main(query:str, unique_id : str):
            config = {"configurable":{"thread_id": unique_id}}
            inputs = {
                "messages": [HumanMessage(query)], 
            }
            async for output in async_app.astream(inputs, config):
                for key, value in output.items():
                    if key != "database_query":
                        yield value['messages'][0].content 
                    else:
                        for response in value['messages']:
                            print(response.content)
                        yield value['generation']

        prev = None
        generation = None
        async for result in main(query, unique_id):
            if prev != None:
                print(prev)
            prev = result
            generation = prev
        return generation

        # curr = None
        # generation = None
        # async for result in main(query):
        #     if curr != None:
        #         st.write(curr)
        #         if "messages" in st.session_state:
        #             st.session_state.messages.append({"role": "assistant", "content": curr})
        #     curr = generation
        #     generation = result
        # return generation
            


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
    
llm = GAMER()

# async def main():
#     query = "How many records are in the database?"
#     result = await llm.streamlit_astream(query, unique_id = "1")
#     print(result)


# asyncio.run(main())

# async def main():
#     result = await llm.ainvoke("How many records are in the database?")
#     print(result)

# asyncio.run(main())