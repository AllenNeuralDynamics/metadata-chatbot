"""Evaluator for GAMER's async workflow"""

from evaluation_dataset import dataset_name
from langchain_core.prompts.prompt import PromptTemplate
from langsmith import aevaluate
from langsmith.evaluation import LangChainStringEvaluator
from utils import LLM

from metadata_chatbot.agents.GAMER import GAMER

_PROMPT_TEMPLATE = """
You are an expert professor specialized in grading students'
answers to questions.
You are grading the following question:
{query}
Here is the real answer:
{answer}
You are grading the following predicted answer:
{result}
Respond with CORRECT or INCORRECT:
Grade:
"""

PROMPT = PromptTemplate(
    input_variables=["query", "answer", "result"], template=_PROMPT_TEMPLATE
)

evaluator = LangChainStringEvaluator(
    "qa", config={"llm": LLM, "prompt": PROMPT}
)


async def my_app(question):
    """Invoking GAMER"""
    model = GAMER()
    return await model.ainvoke(question)


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
        experiment_prefix="async-metadata-chatbot-0.0.65",
    )
    return experiment_results


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
