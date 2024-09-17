import boto3, json, os, logging
from tools import doc_retrieval, projection_retrieval, aggregation_retrieval, tool_call
from system_prompt import system_prompt, summary_system_prompt
from config import toolConfig
from botocore.exceptions import ClientError

logging.basicConfig(filename='error.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#Connecting to bedrock

bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name = 'us-west-2'
)

model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

def get_completion(prompt: str, bedrock_client, system_prompt=system_prompt) -> str:
    
    """Given a prompt, this function returns a reply to the question.

    Parameters
    ----------
    prompt: str 
        Query given by the user to the chatbot
    bedrock_client: variable
        Initialization of boto3 bedrock client
    system_prompt: str 
        Commands to be given to the model, will determine model response
    prefill: str 
        Formatted prefill words to start Claude's reply

    Returns
    -------
    str
        Model's reply to prompt
    """

    messages = [{"role": "user", "content": [{"text": prompt}]}]
    
    inference_config = {
        "temperature": 0,
        "maxTokens": 4096
    }
    converse_api_params = {
        "modelId": model_id,
        "messages" : messages,
        "inferenceConfig": inference_config,
        "toolConfig": toolConfig,
        "system" : [{"text": system_prompt}] 
    }

    try:
        response = bedrock_client.converse(**converse_api_params)  
        print(response)
        response_content_blocks = response['output']['message']['content']

        #Printing Claude's initial response to query
        print(response_content_blocks[0]['text'])
        
        #Assistant reply including tool use 
        messages.append({"role": "assistant", "content": response_content_blocks})

        if response['stopReason'] == "tool_use":
            tool_use = response_content_blocks[-1]['toolUse']
            tool_id = tool_use['toolUseId']
            tool_name = tool_use['name']
            tool_inputs = tool_use['input']
            
            logging.info(f"Using tool {tool_name}")
            
            retrieved_info = tool_call(tool_name, tool_inputs)

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
            logging.info("Successful information retrieval")
                
            converse_api_params = {
                                    "modelId": model_id,
                                    "messages": messages,
                                    "inferenceConfig": inference_config,
                                    "toolConfig": toolConfig 
                                    }
            
            logging.info("Generating response...")
            final_response = bedrock_client.converse(**converse_api_params) 
            final_response_text = final_response['output']['message']['content'][0]['text']
            return(final_response_text)
                    
    except ClientError as e:
        logging.error("A client exception occurred: %s", str(e), exc_info=True)


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
                
                if tool_name == 'doc_retrieval':
                    filter_query_s = tool_inputs['filter'] # filter query stored as a string instead of dictionary
                    filter_query = json.loads(filter_query_s)
                    retrieved_info_list = doc_retrieval(filter_query) #retrieved info type, dictionary
                    retrieved_info = " ".join(map(str, retrieved_info_list))
             
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

                final_response = bedrock_client.converse(**converse_api_params) 
                final_response_text = final_response['output']['message']['content'][0]['text']
                return(final_response_text)
                    
    except ClientError as err:
        message = err.response['Error']['Message']
        print(f"A client error occured: {message}")
        
        
def simple_chat(bedrock_client = bedrock, system_prompt = system_prompt):
    
    """This function is able to demonstrate back and forth conversation given user input.

    Parameters
    ----------
    bedrock_client: variable
        Initialization of boto3 bedrock client
    system_prompt: str 
        Commands to be given to the model, will determine model response

    Returns
    -------
    str
        Model's reply to prompt
    """

    model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
    
    user_message = input("\nUser: ")
    messages = [{"role": "user", "content": [{"text": user_message}]}]
    
    inference_config = {
        "temperature": 0,
        "maxTokens": 4000
    }
    
    while True:
        #If the last message is from the assistant, get another input from the user
        if messages[-1].get("role") == "assistant":
            user_message = input("\nUser: ")
            messages.append({"role": "user", "content": [{"text": user_message}]})

        converse_api_params = {
            "modelId": model_id,
            "messages": messages,
            "inferenceConfig": inference_config,
            "toolConfig":toolConfig,
        }
        if system_prompt:
            converse_api_params["system"] = [{"text": system_prompt}]

        response = bedrock_client.converse(**converse_api_params)
        print(response)

        messages.append({"role": "assistant", "content": response['output']['message']['content']})

        #If Claude stops because it wants to use a tool:
        if response['stopReason'] == "tool_use":
            tool_use = response['output']['message']['content'][-1] #Naive approach assumes only 1 tool is called at a time
            tool_id = tool_use['toolUse']['toolUseId']
            tool_name = tool_use['toolUse']['name']
            tool_inputs = tool_use['toolUse']['input']

            print(f"Using the {tool_name} tool...")
            print(f"Tool Input:")
            print(json.dumps(tool_inputs, indent=2))
            
            retrieved_info = tool_call(tool_name, tool_inputs)
            

            messages.append({
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
                        
                        }
                    }
                ]
            })

        else: 
            print("\nClaude:" + f"{response['output']['message']['content'][0]['text']}")
            #print("\nClaude: Is there anything else I can help you with?")

if __name__ == '__main__': 
    #simple_chat(bedrock)
    prompt = "How many experiments of each unique modality exists in the database?"
    response = get_completion(prompt, bedrock)
    print(response)