import asyncio
import time

from langchain_core.messages import HumanMessage

from metadata_chatbot.evaluations.bedrock.mongodb_bedrock import bedrock_app
from metadata_chatbot.evaluations.benchmarks import mongodb_benchmark, mongodb_python_benchmark
from metadata_chatbot.models import evaluator_chain

benchmark = mongodb_python_benchmark

async def main():
    for index, row in benchmark.iterrows():

        response = "Error occurred"
        mongodb_query = None
        time_taken = -1
        mongodb_evaluation = "ERROR"
        mongodb_score = 0
        response_evaluation = "ERROR"
        response_score = 0
        tool_output_size = 0

        query = row["input_question"]
        target_response = row["target_answer"]
        target_mongodb_query = row["target_mongodb_query"]
        inputs = {"messages": [HumanMessage(query)], "query": query}

        try:
            try:
                start = time.time()
                answer = await bedrock_app.ainvoke(inputs)
                end = time.time()
                time_taken = end - start
                response = answer["generation"]
                mongodb_query = answer["mongodb_query"]["tool_call_0"]["args"]
                tool_output_size = answer["tool_output_size"]

            except Exception as e:
                response = f"Error: {e}"
                mongodb_query = f"Error: {e}"

            benchmark.at[index, "predicted_answer"] = response
            benchmark.at[index, "predicted_mongodb_query"] = (
                mongodb_query
            )
            benchmark.at[index, "generation_time"] = time_taken
            benchmark.at[index, "tool_output_size"] = tool_output_size 

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

        benchmark.at[index, "response_evaluation"] = (
            response_evaluation
        )
        benchmark.at[index, "response_score"] = response_score
        benchmark.at[index, "mongodb_evaluation"] = mongodb_evaluation
        benchmark.at[index, "mongodb_score"] = mongodb_score

    return benchmark


if __name__ == "__main__":
    results = asyncio.run(main())

    results.to_csv("bedrock_mongodb_evals.csv", index=False)

    print(
        f"Processed {len(results)} queries and saved results to bedrock_mongodb_eval_results.json"
    )
