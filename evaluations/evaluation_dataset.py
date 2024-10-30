from langsmith import Client

client = Client()

# Define dataset: these are your test cases
dataset_name = "RAG testing example Dataset"
'''
dataset = client.create_dataset(dataset_name)
client.create_examples(
    inputs=[
        {"question": "What is the genotype for subject 675387?"},
        {"question": "What is the genotype for the subject in SmartSPIM_675387_2023-05-23_23-05-56?"},
        {"question": "What is the genotype for mouse 675387?"},
        {"question": "What are the injections for SmartSPIM_675387_2023-05-23_23-05-56?"},
        {"question": "What are all the assets using mouse 675387"},
        {"question": "Write a MongoDB query to find the genotype of SmartSPIM_675387_2023-05-23_23-05-56"},
        {"question": "How many records are stored in the database?"},
        {"question": "What are the unique modalities found in the database?"},
    ],
    outputs=[
        {"answer": "The genotype for subject 675387 is wt/wt"},
        {"answer": "The genotype for the subject in SmartSPIM_675387_2023-05-23_23-05-56 is wt/wt"},
        {"answer": "The genotype for mouse 675387 is wt/wt"},
        {"answer":  """
                    Based on the provided context, the injections for SmartSPIM_675387_2023-05-23_23-05-56 are:
                    1. AAVrg-Syn-H2B-Turquoise virus with titer 48000000000000 gc/mL, injected at coordinates ML 1.0 mm, AP 1.2 mm, depth 0.8 mm from Bregma, with injection volume 50.0 nanoliters, targeting the Isocortex in the right hemisphere.
                    2. AAVrg-Syn-H2B-tdTomato virus with titer 51000000000000 gc/mL, injected at coordinates ML 1.4 mm, AP 1.2 mm, depth 0.8 mm from Bregma, with injection volume 50.0 nanoliters, targeting the Isocortex in the right hemisphere.
                    Both injections were performed using the Nanoject injection procedure with a 5.0 minute recovery time.
                    """},
        {"answer": "SmartSPIM_675387_2023-05-23_23-05-56 and SmartSPIM_675387_2023-05-23_23-05-56_stitched_2023-06-01_22-30-44"},
        {"answer":  """
                    <query>{
                            "$match": {"name":"SmartSPIM_675387_2023-05-23_23-05-56"},
                            "$project": {"_id": 0, "genotype": "$subject.genotype"}
                    }</query>
                    To find the genotype for the experiment with the name "SmartSPIM_675387_2023-05-23_23-05-56", the MongoDB query would be:
                    1. The `$match` stage filters the documents to only include the one with the specified name.
                    2. The `$project` stage excludes the `_id` field and includes the `genotype` field from the nested `subject` object.
                    The retrieved output shows that the genotype for this experiment is "wt/wt".
                    """},
        {"answer": "There are 267 records found in the database."},
        {"answer": "The unique modalities in the database are Behavior, Behavior videos, Planar optical physiology and Selective plane illumination microscopy."},
    ],
    dataset_id=dataset.id,
)
'''
