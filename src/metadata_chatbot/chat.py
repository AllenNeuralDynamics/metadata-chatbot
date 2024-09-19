import boto3, json, os, logging
from tools import doc_retrieval
from system_prompt import summary_system_prompt
from config import toolConfig
from botocore.exceptions import ClientError

logging.basicConfig(filename='error.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#Connecting to bedrock

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name = 'us-west-2'
)

model_id = "anthropic.claude-3-sonnet-20240229-v1:0"


def get_summary(prompt, bedrock_client = bedrock, system_prompt=summary_system_prompt):

    messages = [{"role": "user", "content": [{"text": f"Summarize the record with id {prompt}"}]}]
    
    inference_config = {
        "temperature": 0,
        "maxTokens": 2000
    }
    converse_api_params = {
        "modelId": model_id,
        "messages" : messages,
        "inferenceConfig": inference_config,
        "toolConfig": toolConfig
    }
    
    if system_prompt:
        converse_api_params["system"] = [{"text": system_prompt}]

    try:
        logging.info("Connecting to model...")
        response = bedrock_client.converse(**converse_api_params)
        
        response_message = response['output']['message']
        
        response_content_blocks = response_message['content']
        
        messages.append({"role": "assistant", "content": response_content_blocks})
        
        for content_block in response_content_blocks:
            if 'toolUse' in content_block:
                
                tool_use = response_content_blocks[-1]
                tool_id = tool_use['toolUse']['toolUseId']
                tool_name = tool_use['toolUse']['name']
                tool_inputs = tool_use['toolUse']['input']

                logging.info("Connecting to database...")

                if tool_name == 'doc_retrieval':
                    filter_query_s = tool_inputs['filter'] # filter query stored as a string instead of dictionary
                    filter_query = json.loads(filter_query_s)
                    retrieved_info_list = doc_retrieval(filter_query) #retrieved info type, dictionary
                    retrieved_info = " ".join(map(str, retrieved_info_list))

                logging.info("Information retrieved from database.")

                tool_response = {
                                "role": "user",
                                "content": [
                                            {
                                            "toolResult": {
                                                "toolUseId": tool_id,
                                                "content": [
                                                    {
                                                    "text": retrieved_info
                                                    }
                                                ],
                                            'status':'success'
                                            }
                                        }
                                        ]
                                }
                    
                messages.append(tool_response)
                    
                converse_api_params = {
                                        "modelId": model_id,
                                        "messages": messages,
                                        "inferenceConfig": inference_config,
                                        "toolConfig": toolConfig 
                                        }
                
                logging.info("Generating summary...")

                final_response = bedrock_client.converse(**converse_api_params) 
                final_response_text = final_response['output']['message']['content'][0]['text']
                #print(final_response_text)
                return(final_response_text)
                    
    except ClientError as err:
        message = err.response['Error']['Message']
        print(f"A client error occured: {message}")
    
if __name__ == '__main__': 
    prompt = "cd2acb6f-e71e-4a5d-8045-a200571950bb"
    print(get_summary(prompt))