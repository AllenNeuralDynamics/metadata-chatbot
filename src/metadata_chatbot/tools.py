from aind_data_access_api.document_db import MetadataDbClient
from aind_data_access_api.document_db_ssh import DocumentDbSSHClient

API_GATEWAY_HOST = "api.allenneuraldynamics.org"
DATABASE = "metadata_index"
COLLECTION = "data_assets"

docdb_api_client = MetadataDbClient(
   host=API_GATEWAY_HOST,
   database=DATABASE,
   collection=COLLECTION,
)

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

def projection_retrieval(credentials: object, filter_query: dict, field_name_list: list):
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

    with DocumentDbSSHClient(credentials=credentials) as doc_db_client:
        filter = filter_query
        projection = {"name" : 1}
        if field_name_list:
            for field_name in field_name_list:
                projection[field_name] = 1
        #count = doc_db_client.collection.count_documents(filter)
        response = list(doc_db_client.collection.find(filter=filter, projection=projection))        
    return response

def get_image(image_path):
    with open(image_path, "rb") as f:
        image_file = f.read()

    return image_file