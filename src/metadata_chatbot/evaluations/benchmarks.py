import pandas as pd

benchmark = pd.read_csv(
    r"C:\Users\sreya.kumar\Documents\GitHub\metadata-chatbot\src\metadata_chatbot\evaluations\benchmark_files\eval_benchmark_4_14_25.csv"
)

# Combining answer columns
benchmark["answer"] = pd.concat(
    [benchmark["output_answer"], benchmark["output_answer.1"]]
).dropna()
# Dropping columns used to combine
benchmark = benchmark.drop(columns=["output_answer", "output_answer.1"])
benchmark = benchmark.rename(
    columns={
        "answer": "target_answer",
        "output_mongodb_query": "target_mongodb_query",
        "output_python": "target_python",
    }
)
benchmark["predicted_answer"] = pd.Series(dtype="str")
benchmark["data_source"] = pd.Series(dtype="str")
benchmark["generation_time"] = pd.Series(dtype="float")
benchmark["response_evaluation"] = pd.Series(dtype="str")
benchmark["response_score"] = pd.Series(dtype="int")

benchmark["predicted_mongodb_query"] = pd.Series(dtype="str")
benchmark["mongodb_evaluation"] = pd.Series(dtype="str")
benchmark["mongodb_score"] = pd.Series(dtype="int")

test_benchmark = benchmark.head(5)

mongodb_benchmark = benchmark.dropna(subset=["target_mongodb_query"])
mongodb_python_benchmark = mongodb_benchmark.dropna(subset=["target_python"])
mongodb_python_benchmark["tool_output_size"] = pd.Series(dtype="int")
mongodb_python_benchmark = mongodb_python_benchmark.reset_index(drop=True)

mongodb_benchmark = mongodb_benchmark.drop(columns=["target_python"])
mongodb_benchmark = mongodb_benchmark.reset_index(drop=True)

python_benchmark = benchmark.dropna(subset=["target_python"])
python_benchmark["predicted_python"] = pd.Series(dtype="str")
python_benchmark["python_evaluation"] = pd.Series(dtype="str")
python_benchmark["python_score"] = pd.Series(dtype="int")
python_benchmark["tool_output_size"] = pd.Series(dtype="int")
python_benchmark = python_benchmark.drop(columns=["data_source",
                                                  'predicted_mongodb_query',
                                                  'mongodb_evaluation', 
                                                  'mongodb_score'])

python_benchmark = python_benchmark.reset_index(drop=True)
