import boto3, json, os
from tools import doc_retrieval, projection_retrieval
from system_prompt import system_prompt
from config import toolConfig
from botocore.exceptions import ClientError

#Connecting to bedrock

client = boto3.client("bedrock-runtime", region_name="us-west-2")
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name = 'us-west-2'
)

def get_completion(prompt, system_prompt=system_prompt, prefill=None):
    
    '''
    Given a prompt, this function returns a reply to the question.
    
    :param prompt: String formatted question
    :param system_prompt: String formatted system prompt 
    :param prefill: String formatted prefill words to start Claude's reply
    
    :return: String formatted answer
    '''
    
    messages = [{"role": "user", "content": [{"text": prompt}]}]
    
    inference_config = {
        "temperature": 0,
        "maxTokens": 4000
    }
    converse_api_params = {
        "modelId": model_id,
        "messages" : messages,
        "inferenceConfig": inference_config,
        "toolConfig": toolConfig
    }
    
    if system_prompt:
        converse_api_params["system"] = [{"text": system_prompt}]
        
    if prefill:
        messages.append({"role": "assistant", "content": [{"text": prefill}]})
        print(prefill)
        

    try:
        response = bedrock.converse(**converse_api_params)
        #print(response)
        
        response_message = response['output']['message']
        
        response_content_blocks = response_message['content']
        
        #Assistant reply including tool use 
        messages.append({"role": "assistant", "content": response_content_blocks})
        
        for content_block in response_content_blocks:
            if 'toolUse' in content_block:
                #print("Stop Reason:", response['stopReason'])
                
                tool_use = response_content_blocks[-1]
                tool_id = tool_use['toolUse']['toolUseId']
                tool_name = tool_use['toolUse']['name']
                tool_inputs = tool_use['toolUse']['input']

                #tool_use_block = content_block['toolUse']
                #tool_use_name = tool_use_block['name']
                
                print(f"Using tool {tool_name}")
                
                if tool_inputs['filter']:
                    filter_query_s = tool_inputs['filter'] # filter query stored as a string instead of dictionary
                    filter_query = json.loads(filter_query_s)
                
                if tool_name == 'doc_retrieval':
                    retrieved_info = doc_retrieval(filter_query) #retrieved info type, dictionary
                    if type(retrieved_info) == list:
                        retrieved_info = {item['_id']:item for item in retrieved_info}
                        
                elif tool_name == 'projection_retrieval':
                    field_name_list = (tool_inputs['fieldNameList'])
                    retrieved_info_list = projection_retrieval(filter_query, field_name_list)
                    retrieved_info = json.dumps(retrieved_info_list)[:1000]

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

                    final_response = bedrock.converse(**converse_api_params) 
                    #print(final_response)
                    final_response_text = final_response['output']['message']['content'][0]['text']
                    return(final_response_text)
                    
                    #eturn messages
                    
                    #return retrieved_info
                    
                #return messages
                
        
        #return response_message
        #return messages

        
    except ClientError as err:
        message = err.response['Error']['Message']
        print(f"A client error occured: {message}")
        
        
def simple_chat(system_prompt = system_prompt):
    
    '''
    This function is able to take user input and provide a reply. Follows a typical chatbot format, where user can ask multiple questions in one iteration.
    
    :param system_prompt: String formatted system prompt 
    
    :return: String formatted answer
    '''
    
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
            

        response = bedrock.converse(**converse_api_params)

        messages.append({"role": "assistant", "content": response['output']['message']['content']})

        #If Claude stops because it wants to use a tool:
        if response['stopReason'] == "tool_use":
            tool_use = response['output']['message']['content'][-1] #Naive approach assumes only 1 tool is called at a time
            tool_id = tool_use['toolUse']['toolUseId']
            tool_name = tool_use['toolUse']['name']
            tool_inputs = tool_use['toolUse']['input']

            print(f"Claude wants to use the {tool_name} tool")
            print(f"Tool Input:")
            print(json.dumps(tool_inputs, indent=2))
            
            if tool_inputs['filter']:
                filter_query_s = tool_inputs['filter'] # filter query stored as a string instead of dictionary
                filter_query = json.loads(filter_query_s)
                
            if tool_name == 'doc_retrieval':
                retrieved_info = doc_retrieval(filter_query) #retrieved info type, dictionary
                if type(retrieved_info) == list:
                    retrieved_info = ''.join(str(x) for x in retrieved_info) #tool response expects a string
                #print('I AM INPUTTING DOC RETRIEVAL INTO RESPONSE')
                        
            elif tool_name == 'projection_retrieval':
                field_name_list = (tool_inputs['fieldNameList'])
                retrieved_info_list = projection_retrieval(filter_query, field_name_list)
                retrieved_info = json.dumps(retrieved_info_list)[:1000]
                #print('I AM INPUTTING PROJECTION RETRIEVAL INTO RESPONSE')

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
            print("\nClaude: " + f"{response['output']['message']['content'][0]['text']}")