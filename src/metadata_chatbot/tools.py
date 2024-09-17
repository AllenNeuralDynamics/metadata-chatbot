from aind_data_access_api.document_db import MetadataDbClient
from aind_data_access_api.document_db_ssh import DocumentDbSSHClient, DocumentDbSSHCredentials
import json

API_GATEWAY_HOST = "api.allenneuraldynamics.org"
DATABASE = "metadata_index"
COLLECTION = "data_assets"

docdb_api_client = MetadataDbClient(
   host=API_GATEWAY_HOST,
   database=DATABASE,
   collection=COLLECTION,
)

#credentials = DocumentDbSSHCredentials()

def doc_retrieval(filter_query: dict) -> list:
    """Given a MongoDB query, this function retrieves and returns the appropriate documents.

    Parameters
    ----------
    filter_query
        MongoDB query

    Returns
    -------
    list
        List of retrieved documents
    """
    limit = 1000
    paginate_batch_size = 1000
    response = docdb_api_client.retrieve_docdb_records(
       filter_query=filter_query,
       limit=limit,
       paginate_batch_size=paginate_batch_size
    )
    return(response)

def projection_retrieval(filter_query: dict, field_name_list: list) -> list:
    """Given a MongoDB query and list of projections, this function retrieves 
    and returns the appropriate projections in the documents.

    Parameters
    ----------
    credentials 
        DocDB credentials, initialized through DocumentDbSSHCredentials

    filter_query
        MongoDB query

    field_name_list
        Field names to specifically retrieve from documents

    Returns
    -------
    list
        List of retrieved documents
    """
    projection = {"name" : 1}
    if field_name_list:
        for field_name in field_name_list:
            projection[field_name] = 1

    response = docdb_api_client.retrieve_docdb_records(
        filter_query=filter_query,
        projection=projection
    )     
    return response

def aggregation_retrieval(agg_pipeline: list) -> list:
    """Given a MongoDB query and list of projections, this function retrieves and returns the 
    relevant information in the documents. 
    Use a project stage as the first stage to minimize the size of the queries before proceeding with the remaining steps.
    The input to $map must be an array not a string, avoid using it in the $project stage.

    Parameters
    ----------
    agg_pipeline
        MongoDB aggregation pipeline

    Returns
    -------
    list
        List of retrieved documents
    """
    result = docdb_api_client.aggregate_docdb_records(
        pipeline=agg_pipeline
    )
    return result

def tool_call(tool_name:str, tool_inputs:dict) -> str:

    if tool_name == 'doc_retrieval':
        filter_query = json.loads(tool_inputs['filter'])
        retrieved_info_list = doc_retrieval(filter_query) #retrieved info type, dictionary
                    
    elif tool_name == 'projection_retrieval':
        filter_query = json.loads(tool_inputs['filter'])
        field_name_list = json.loads(tool_inputs['fieldNameList'])
        retrieved_info_list = projection_retrieval(filter_query, field_name_list)
        #retrieved_info = json.dumps(retrieved_info_list)[:1000]
                
    elif tool_name == 'aggregation_retrieval':
        #print("Loading agg pipeline...")
        agg_pipeline = json.loads(tool_inputs['pipeline'])
        #print(type(tool_inputs['pipeline']))
        retrieved_info_list = aggregation_retrieval(agg_pipeline)
        #print("Retrieved info ready")
    
    retrieved_info = " ".join(map(str, retrieved_info_list))
    #print(retrieved_info)
    return(retrieved_info)
