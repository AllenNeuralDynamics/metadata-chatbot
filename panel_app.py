import panel as pn
from time import sleep

from metadata_chatbot.agents.async_workflow import async_workflow
from metadata_chatbot.agents.react_agent import astream_input

pn.extension()

model = async_workflow.compile()
messages = []

async def get_response(query: str):
    chat_history = messages
    inputs = {
        "messages": chat_history, 
    }
    async for output in model.astream(inputs):
        for key, value in output.items():
            if key != "database_query":
                yield value['messages'][0].content 
            else:
                try:
                    query = str(chat_history) + query
                    async for result in astream_input(query = query):
                        response = result['type']
                        if response == 'intermediate_steps':
                            yield result['content']
                        if response == 'agg_pipeline':
                            yield f"The MongoDB pipeline used to on the database is: {result['content']}"
                        if response == 'tool_response':
                            yield f"Retrieved output from MongoDB: {result['content']}"
                        if response == 'final_answer':
                            yield result['content']
                except Exception as e:
                    yield f"An error has occured with the retrieval from DocDB: {e}. Try structuring your query another way."
                # for response in value['messages']:
                #     yield response.content
                # yield value['generation
    # prev = None
    # generation = None
    # async for result in get_response(query):
    #     if prev != None:
    #         st.markdown(prev)
    #     prev = result
    #     generation = prev
    # st.markdown(generation)

# def get_response(contents, user, instance):
#     if "turbine" in contents.lower():
#         response = "A wind turbine converts wind energy into electricity."
#     else:
#         response = "Sorry, I don't know."
#     for index in range(len(response)):
#         yield response[0:index+1]
#         sleep(0.03) # to simulate slowish response

chat_bot = pn.chat.ChatInterface(callback=get_response, max_height=500)
chat_bot.send("Ask me anything!", user="Assistant", respond=False)
chat_bot.servable()