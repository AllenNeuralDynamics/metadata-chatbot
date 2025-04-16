import asyncio
import time

from langchain_core.messages import HumanMessage

from metadata_chatbot.evaluations.bedrock.workflow import app
from metadata_chatbot.evaluations.benchmarks import benchmark, test_benchmark
from metadata_chatbot.models import evaluator_chain, evaluator_python_chain

async def main():
    for index, row in benchmark.iterrows():

        query = row["input_question"]
        target_response = row["target_answer"]
        target_mongodb_query = row["target_mongodb_query"]
        target_python = row["target_mongodb_query"]
        inputs = {"messages": [HumanMessage(query)], "query": query}

        response = "Error occurred"
        mongodb_query = None
        time_taken = -1
        mongodb_evaluation = "ERROR"
        mongodb_score = 0
        response_evaluation = "ERROR"
        response_score = 0
        tool_output_size = 0

        try:
            start = time.time()
            answer = await app.ainvoke(inputs)
            end = time.time()
            time_taken = end - start
            response = answer.get("generation", "No response generated")
            data_source = answer.get("data_source", "No data source used")

            mongodb_query = None
            if "mongodb_query" in answer and answer["mongodb_query"]:
                if "tool_call_0" in answer["mongodb_query"]:
                    mongodb_query = answer["mongodb_query"]["tool_call_0"].get("args", "No arguments found")
                    tool_output_size = answer["tool_output_size"]
                else:
                    mongodb_query = "MongoDB query present but in different format"
                    tool_output_size = 0
            else:
                mongodb_query = "No MongoDB query was generated"
                tool_output_size = 0
            if mongodb_query:
                benchmark.at[index, "predicted_mongodb_query"] = mongodb_query
                benchmark.at[index, "tool_output_size"] = tool_output_size 
                
                # Evaluate MongoDB query
                try:
                    mongodb_result = await evaluator_chain.ainvoke({
                        "query": query,
                        "target": target_mongodb_query,
                        "predicted": mongodb_query,
                    })
                    
                    mongodb_evaluation = mongodb_result.get("score", "ERROR")
                    mongodb_score = 1 if mongodb_evaluation == "CORRECT" else 0
                except Exception as e:
                    print(f"MongoDB evaluation error: {e}")
                    mongodb_evaluation = f"ERROR: {str(e)[:100]}"
                    mongodb_score = 0
            else:
                benchmark.at[index, "predicted_mongodb_query"] = "nan"
                mongodb_evaluation = "nan"
                mongodb_score = 0
            benchmark.at[index, "mongodb_evaluation"] = mongodb_evaluation
            benchmark.at[index, "mongodb_score"] = mongodb_score
            benchmark.at[index, "predicted_answer"] = response
    
            benchmark.at[index, "generation_time"] = time_taken
            benchmark.at[index, "data_source"] = data_source

            
            # Evaluate response
            try:
                response_result = await evaluator_python_chain.ainvoke({
                    "query": query,
                    "target": target_response,
                    "predicted": response,
                    "python_code": target_python,
                })
                
                response_evaluation = response_result.get("score", "ERROR")
                response_score = 1 if response_evaluation == "CORRECT" else 0
            except Exception as e:
                print(f"Response evaluation error: {e}")
                response_evaluation = f"ERROR: {str(e)[:100]}"
                response_score = 0
                
        except Exception as e:
            print(f"Main pipeline error: {e}")
            response = f"Error: {str(e)[:100]}"
            benchmark.at[index, "predicted_answer"] = response
            benchmark.at[index, "predicted_mongodb_query"] = "ERROR"
        
        # Always update these fields regardless of exceptions
        benchmark.at[index, "response_evaluation"] = response_evaluation
        benchmark.at[index, "response_score"] = response_score
        
    return benchmark
        




if __name__ == "__main__":

    results = asyncio.run(main())
    results.to_csv("bedrock_evals.csv", index=False)

    print(
        f"Processed {len(results)} queries and saved results to bedrock_evals.csv"
    )
