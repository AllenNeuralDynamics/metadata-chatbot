"""Instantiating GAMER call"""

import asyncio
import uuid
from typing import Any, Dict, Iterator, List, Optional

from langchain.callbacks.manager import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import LLM
from langchain_core.messages import HumanMessage
from langchain_core.outputs import GenerationChunk

from metadata_chatbot.agents.async_workflow import async_app


class GAMER(LLM):
    """Class for metadata querying tool"""

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
            stop: Stop words to use when generating. Model output
                is cut off at the first occurrence of any of the
                stop substrings.If stop tokens are not supported
                consider raising NotImplementedError.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments.
                These are usually passedto the model provider API call.

        Returns:
            The model output as a string.
        """
        asyncio.run(self._acall(query))

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
            """Streaming content within each node in GAMER"""

            unique_id = str(uuid.uuid4())
            config = {"configurable": {"thread_id": unique_id}}
            inputs = {
                "messages": [HumanMessage(query)],
            }
            async for output in async_app.astream(inputs, config):
                for key, value in output.items():
                    if key != "database_query":
                        yield value["messages"][0].content

        curr = None
        generation = None
        async for result in main(query):
            if curr is not None:
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
        """Stream the LLM on the given prompt."""
        for char in query[: self.n]:
            chunk = GenerationChunk(text=char)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

            yield chunk

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            "model_name": "Anthropic Claude 3 Sonnet",
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model.
        Used for logging purposes only."""
        return "Claude 3 Sonnet"


llm = GAMER()


# async def main():
#     result = await llm.ainvoke("How many records are in the database?")
#     print(result)

# asyncio.run(main())
