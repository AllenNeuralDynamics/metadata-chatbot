# import pytest, boto3
# from langsmith import unit, expect
# from metadata_chatbot.agents.agentic_graph import datasource_router, aggregation_retrieval, query_retriever, db_rag_chain
# from langchain_aws import BedrockEmbeddings
# BEDROCK_CLIENT = boto3.client(
#     service_name="bedrock-runtime",
#     region_name = 'us-west-2'
# )

# BEDROCK_EMBEDDINGS = BedrockEmbeddings(model_id="amazon.titan-embed-text-v2:0",client=BEDROCK_CLIENT)

# @pytest.fixture
# def user_query():
#     return "Write a MongoDB query to find the genotype of SmartSPIM_675387_2023-05-23_23-05-56"

# @unit
# def test_datasource_router_database():
#     datasource = datasource_router.invoke({"query": user_query}).datasource
#     assert datasource == "direct_database"

# @unit
# @pytest.mark.parametrize(
#     "user_query, expected_output",
#     [
#         ("Write a MongoDB query to find the genotype of SmartSPIM_675387_2023-05-23_23-05-56", 
#          """
#         {\"mongodb_query\": [{\"$match\": {\"name\": \"SmartSPIM_675387_2023-05-23_23-05-56\"}}, 
#         {\"$project\": {\"_id\": 0, \"genotype\": \"$subject.genotype\"}}], \"retrieved_output\": [{\"genotype\": \"wt/wt\"}]}
#         """
#         ),
#         ("What is the genotype for subject 675387?",
#          """
#          {\"mongodb_query\": [{\"$match\": {\"subject.subject_id\": \"675387\"}}, {\"$project\": {\"subject.genotype\": 1, \"_id\": 0}}], 
#          "retrieved_output\": [{\"subject\": {\"genotype\": \"wt/wt\"}}, {\"subject\": {\"genotype\": \"wt/wt\"}}]}
#          """
#         )
#     ]
# )
# def test_query_retriever(user_query, expected_output):
#     prediction = query_retriever.invoke({'chat_history': [],"query": user_query, "agent_scratchpad":[]})
#     expect.embedding_distance(
#         prediction, expected_output, config={"encoder": BEDROCK_EMBEDDINGS, "metric": "cosine"}
#     ).to_be_less_than(0.5)
#     expect.edit_distance(
#         prediction, expected_output
#     )

# @unit
# @pytest.mark.parametrize(
#     "user_query, documents, expected_output",
#     [
#         ("Write a MongoDB query to find the genotype of SmartSPIM_675387_2023-05-23_23-05-56", 
#          """
#         {\"mongodb_query\": [{\"$match\": {\"name\": \"SmartSPIM_675387_2023-05-23_23-05-56\"}}, 
#         {\"$project\": {\"_id\": 0, \"genotype\": \"$subject.genotype\"}}], \"retrieved_output\": [{\"genotype\": \"wt/wt\"}]}
#          """,
#          """
#         <query>{\n  \"$match\": {\n    \"name\": \"SmartSPIM_675387_2023-05-23_23-05-56\"\n  },\n  \"$project\": {\n    \"_id\": 0,\n    \"genotype\": \"$subject.genotype\"\n  }\n}</query>
#         To find the genotype for the experiment with the name \"SmartSPIM_675387_2023-05-23_23-05-56\", the MongoDB query would be:
#         1. The `$match` stage filters the documents to only include the one with the specified name.
#         2. The `$project` stage excludes the `_id` field and includes the `genotype` field from the nested `subject` object.
#         The retrieved output shows that the genotype for this experiment is \"wt/wt\".
#         """
#         ),
#         ("What is the genotype for subject 675387?",
#          """
#         {\"mongodb_query\": [{\"$match\": {\"subject.subject_id\": \"675387\"}}, {\"$project\": {\"subject.genotype\": 1, \"_id\": 0}}], 
#          "retrieved_output\": [{\"subject\": {\"genotype\": \"wt/wt\"}}, {\"subject\": {\"genotype\": \"wt/wt\"}}]}
#          """,
#         "The genotype for subject 675387 is wt/wt.")
#     ]
# )
# def test_db_rag_chain(user_query, documents, expected_output):
#     prediction = db_rag_chain.invoke({"query": user_query, 
#                                       "documents": documents})
#     expect.embedding_distance(
#         prediction, expected_output, config={"encoder": BEDROCK_EMBEDDINGS, "metric": "cosine"}
#     ).to_be_less_than(0.5)
#     expect.edit_distance(
#         prediction, expected_output
#     )