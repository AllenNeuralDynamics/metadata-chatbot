import asyncio
import time

from langchain_core.messages import HumanMessage

from metadata_chatbot.evaluations.anthropic.python_gen import (
    app,
)
from metadata_chatbot.evaluations.benchmarks import python_benchmark
from metadata_chatbot.models import evaluator_chain, evaluator_python_chain

benchmark = python_benchmark

async def main():
    for index, row in benchmark.iterrows():
        response_evaluation = "ERROR"
        python_evaluation = "ERROR"
        response = "ERROR"
        python = "ERROR"
        response_score = 0
        time_taken = 0

        query = row["input_question"]
        target_response = row["target_answer"]
        target_python = row["target_python"]
        inputs = {"messages": [HumanMessage(query)], "query": query}

        try:
            try:
                start = time.time()
                answer = await app.ainvoke(inputs)
                end = time.time()
                time_taken = end - start
                response = answer["generation"]
                python = answer["python_code"]

            except Exception as e:
                response = f"Error: {e}"
                python = f"Error: {e}"

            benchmark.at[index, "predicted_answer"] = response
            benchmark.at[index, "predicted_python"] = (
                python
            )
            benchmark.at[index, "generation_time"] = time_taken

            python_result = await evaluator_python_chain.ainvoke(
                {
                    "query": query,
                    "target": target_python,
                    "predicted": python,
                    "python_code": python
                }
            )

            python_evaluation = python_result["score"]

            response_result = await evaluator_chain.ainvoke(
                {
                    "query": query,
                    "target": target_response,
                    "predicted": response,
                }
            )
            response_evaluation = response_result["score"]

            response_score = 0
            python_score = 0

            if response_evaluation == "CORRECT":
                response_score = 1
            if python_evaluation == "CORRECT":
                python_score = 1

        except Exception as e:
            response_score = f"Error: {e}"
            python_score = f"Error: {e}"

        benchmark.at[index, "response_evaluation"] = (
            response_evaluation
        )
        benchmark.at[index, "response_score"] = response_score
        benchmark.at[index, "python_evaluation"] = python_evaluation
        benchmark.at[index, "python_score"] = python_score

        await asyncio.sleep(20)

    return benchmark



if __name__ == "__main__":
    results = asyncio.run(main())

    results.to_csv("anthropic_python_evals.csv", index=False)

    print(
        f"Processed {len(results)} queries and saved results to anthropic_python_evals.csv"
    )
