"""Evaluator for GAMER's async workflow"""

from langchain_core.prompts.prompt import PromptTemplate
from langsmith import aevaluate
from langsmith.evaluation import LangChainStringEvaluator
from metadata_chatbot.utils import SONNET_3_7_LLM

from metadata_chatbot.GAMER.workflow import app, stream_response

from langchain_core.messages import HumanMessage

dataset_name = "RAG testing example Dataset"

_PROMPT_TEMPLATE = """
You are an expert professor specialized in grading students'
answers to questions.
You are grading the following question:
{query}
Here is the real answer:
{answer}
You are grading the following predicted answer:
{result}
Keep in mind that the real answer and predicted answer might not be the same.
The predicted answers are generated based on a database that undergoes changes everyday.
If the numbers are different, that should be okay. Ensure that any mongodb query generated makes sense.
Respond with CORRECT or INCORRECT:
Grade:
"""

PROMPT = PromptTemplate(
    input_variables=["query", "answer", "result"], template=_PROMPT_TEMPLATE
)

evaluator = LangChainStringEvaluator(
    "qa", config={"llm": SONNET_3_7_LLM, "prompt": PROMPT}
)


async def my_app(query):

    inputs = {
        "messages": [HumanMessage(query)],
    }

    answer = await app.ainvoke(inputs)

    return answer['generation']

async def langsmith_app(inputs):
    """Writing GAMER's generations to langsmith"""
    output = await my_app(inputs["question"])
    return {"output": output}


async def main():
    """Evaluating quality of GAMER's response"""
    experiment_results = await aevaluate(
        langsmith_app,  # Your AI system
        data=dataset_name,  # The data to predict and grade over
        evaluators=[evaluator],  # The evaluators to score the results
        experiment_prefix="GAMER-0.3.8",
    )
    return experiment_results


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
