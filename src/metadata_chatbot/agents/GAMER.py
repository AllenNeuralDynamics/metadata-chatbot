from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk

import logging, asyncio

from metadata_chatbot.agents.async_workflow import async_app
from metadata_chatbot.agents.workflow import app


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
        for output in app.stream(inputs, stream_mode="updates"):
            for key, value in output.items():
                logging.info(f"Currently on node '{key}':")
        return value['generation'] if value else None
        # inputs = {"query" : query}                          
        # answer = app.invoke(inputs)                                           
        # if isinstance(answer, dict):
        #     result_text = answer.get('text', '')  # Adjust as necessary based on the actual structure
        # else:
        #     result_text = answer
        # print(result_text)
        # return answer
    
    async def _acall(
        self,
        query: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Asynchronous call.

        Args:
            query: Natural language query.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of the stop substrings.
                If stop tokens are not supported consider raising NotImplementedError.
            run_manager: Callback manager for the run.

        Returns:
            The model output as a string.
        """
        inputs = {"query" : query}
        async for output in async_app.astream(inputs, stream_mode="updates"):
            for key, value in output.items():
                logging.info(f"Currently on node '{key}':")
        return value['generation'] if value else None

    def _stream(
        self,
        query: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        """Stream the LLM on the given prompt.

        This method should be overridden by subclasses that support streaming.

        If not implemented, the default behavior of calls to stream will be to
        fallback to the non-streaming version of the model and return
        the output as a single chunk.

        Args:
            query: The prompt to generate from.
            stop: Stop words to use when generating. Model output is cut off at the
                first occurrence of any of these substrings.
            run_manager: Callback manager for the run.
            **kwargs: Arbitrary additional keyword arguments. These are usually passed
                to the model provider API call.

        Returns:
            An iterator of GenerationChunks.
        """
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
        """Get the type of language model used by this chat model. Used for logging purposes only."""
        return "Claude 3 Sonnet"
    
# llm = GAMER()
# print(llm.invoke("Can you give me a timeline of events for subject 675387?"))

# async def main():
#     result = await llm.ainvoke("Can you give me a timeline of events for subject 675387?")
#     print(result)

# asyncio.run(main())