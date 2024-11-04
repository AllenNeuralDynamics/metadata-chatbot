from utils import LLM
from langchain_core.prompts.prompt import PromptTemplate
from langsmith.evaluation import LangChainStringEvaluator
from metadata_chatbot.agents.GAMER import GAMER
from langsmith import evaluate
from evaluation_dataset import dataset_name


_PROMPT_TEMPLATE = """You are an expert professor specialized in grading students' answers to questions.
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

evaluator = LangChainStringEvaluator("qa", config={"llm": LLM, "prompt": PROMPT})

def my_app(question):
    model = GAMER()
    return model.invoke(question)

def langsmith_app(inputs):
    output = my_app(inputs["question"])
    return {"output": output}

experiment_results = evaluate(
    langsmith_app, # Your AI system
    data=dataset_name, # The data to predict and grade over
    evaluators=[evaluator], # The evaluators to score the results
    experiment_prefix="metadata-chatbot-0.0.66", # A prefix for your experiment names to easily identify them
)