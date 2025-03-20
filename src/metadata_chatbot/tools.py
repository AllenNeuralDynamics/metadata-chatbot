from aind_data_access_api.document_db import MetadataDbClient
from langchain_core.tools import tool

API_GATEWAY_HOST = "api.allenneuraldynamics.org"
DATABASE = "metadata_index"
COLLECTION = "data_assets"

docdb_api_client = MetadataDbClient(
    host=API_GATEWAY_HOST,
    database=DATABASE,
    collection=COLLECTION,
)


@tool
def aggregation_retrieval(agg_pipeline: list) -> list:
    """
    Executes a MongoDB aggregation pipeline for complex data transformations
    and analysis.

    WHEN TO USE THIS FUNCTION:
    - When you need to perform multi-stage data processing operations
    - For complex queries requiring grouping, filtering, sorting in sequence
    - When you need to calculate aggregated values (sums, averages, counts)
    - For data transformation operations that can't be done with simple queries

    NOT RECOMMENDED FOR:
    - Simple document retrieval (use get_records instead)
    - When you only need to filter data without transformations
    Executes a MongoDB aggregation pipeline and returns the aggregated results.

    This function processes complex queries using MongoDB's aggregation
    framework, allowing for data transformation, filtering, grouping, and
    analysis operations. It handles the execution of multi-stage aggregation
    pipelines and provides error handling for failed aggregations.

    Parameters
    ----------
    agg_pipeline : list
        A list of dictionary objects representing MongoDB aggregation stages.
        Each stage should be a valid MongoDB aggregation operator.
        Common stages include: $match, $project, $group, $sort, $unwind.

    Returns
    -------
    list
        Returns a list of documents resulting from the aggregation pipeline.
        If an error occurs, returns an error message string describing
        the exception.

    Notes
    -----
    - Include a $project stage early in the pipeline to reduce data transfer
    - Avoid using $map operator in $project stages as it requires array inputs
    """
    try:
        result = docdb_api_client.aggregate_docdb_records(
            pipeline=agg_pipeline
        )
        return result

    except Exception as ex:
        template = "An exception of type {0} occurred. Arguments:\n{1!r}"
        message = template.format(type(ex).__name__, ex.args)
        return message


@tool
def get_records(filter: dict = {}, projection: dict = {}) -> dict:
    """
    Retrieves documents from MongoDB database using simple filters
    and projections.

    WHEN TO USE THIS FUNCTION:
    - For straightforward document retrieval based on specific criteria
    - When you need only a subset of fields from documents
    - When the query logic doesn't require multi-stage processing
    - For better performance with simpler queries

    NOT RECOMMENDED FOR:
    - Complex data transformations (use aggregation_retrieval instead)
    - Grouping operations or calculations across documents
    - Joining or relating data across collections

    Parameters
    ----------
    filter : dict
        MongoDB query filter to narrow down the documents to retrieve.
        Example: {"subject.sex": "Male"}
        If empty dict object, returns all documents.

    projection : dict
        Fields to include or exclude in the returned documents.
        Use 1 to include a field, 0 to exclude.
        Example: {"subject.genotype": 1, "_id": 0}
        will return only the genotype field.
        If empty dict object, returns all documents.

    limit: int
        Limit retrievals to a reasonable number, try to not exceed 100

    Returns
    -------
    list
        List of dictionary objects representing the matching documents.
        Each dictionary contains the requested fields based on the projection.

    """

    records = docdb_api_client.retrieve_docdb_records(
        filter_query=filter, projection=projection, limit=100
    )

    return records


tools = [get_records, aggregation_retrieval]
