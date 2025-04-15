import asyncio
import time

from langchain_core.messages import HumanMessage

from metadata_chatbot.evaluations.anthropic.mongodb_anthropic import (
    anthropic_app,
)
from metadata_chatbot.evaluations.benchmarks import mongodb_benchmark
from metadata_chatbot.models import evaluator_chain


async def main():
    for index, row in mongodb_benchmark.iterrows():

        query = row["input_question"]
        target_response = row["target_answer"]
        target_mongodb_query = row["target_mongodb_query"]
        inputs = {"messages": [HumanMessage(query)], "query": query}

        try:
            try:
                start = time.time()
                answer = await anthropic_app.ainvoke(inputs)
                end = time.time()
                time_taken = end - start
                response = answer["generation"]
                mongodb_query = answer["mongodb_query"]["tool_call_0"]["args"]

            except Exception as e:
                response = f"Error: {e}"
                mongodb_query = f"Error: {e}"

            mongodb_benchmark.at[index, "predicted_answer"] = response
            mongodb_benchmark.at[index, "predicted_mongodb_query"] = (
                mongodb_query
            )
            mongodb_benchmark.at[index, "generation_time"] = time_taken

            mongodb_result = await evaluator_chain.ainvoke(
                {
                    "query": query,
                    "target": target_mongodb_query,
                    "predicted": mongodb_query,
                }
            )

            mongodb_evaluation = mongodb_result["score"]

            response_result = await evaluator_chain.ainvoke(
                {
                    "query": query,
                    "target": target_response,
                    "predicted": response,
                }
            )
            response_evaluation = response_result["score"]

            response_score = 0
            mongodb_score = 0

            if response_evaluation == "CORRECT":
                response_score = 1
            if mongodb_evaluation == "CORRECT":
                mongodb_score = 1

        except Exception as e:
            response_score = f"Error: {e}"

        mongodb_benchmark.at[index, "response_evaluation"] = (
            response_evaluation
        )
        mongodb_benchmark.at[index, "response_score"] = response_score
        mongodb_benchmark.at[index, "mongodb_evaluation"] = mongodb_evaluation
        mongodb_benchmark.at[index, "mongodb_score"] = mongodb_score

        await asyncio.sleep(20)

    return mongodb_benchmark



if __name__ == "__main__":
    results = asyncio.run(main())

    mongodb_benchmark.to_csv("anthropic_mongodb_evals.csv", index=False)

    print(
        f"Processed {len(results)} queries and saved results to anthropic_mongodb_eval_results.csv"
    )
