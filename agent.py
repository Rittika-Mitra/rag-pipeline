from langchain.agents import create_openai_functions_agent
from langchain.prompts import PromptTemplate
from langchain.agents import (
    create_openai_functions_agent,
    Tool,
    AgentExecutor,
)

from langchain import hub
from dotenv import load_dotenv
import os 

load_dotenv()  #load all the environment variables
HF_TOKEN = os.getenv("HF_TOKEN")

from chatbot import chat_model, sagemaker_chain

tools = [
    Tool(
        name="sagemakers",
        func=sagemaker_chain.invoke,
        description="""Useful when you need to answer questions
        about sagemakers only.
        Not useful for answering any other questions. For instance,
        if the question is "What are lsit in python?" response should
        be "I don't kno"w as no sagemaker related context in the question.
        """,
    )
]

# Create the agent

agent_prompt = PromptTemplate(
    input_variables=["question", "agent_scratchpad"],
    template="""You are an intelligent agent capable of determining whether the given question is related to Sagemaker or some other topic and routing it accordingly.

Question: {question}

Agent Scratchpad: {agent_scratchpad}

Determine which tool to use for this question and respond accordingly. 
If the question is related to Sagemaker, use the 'sagemakers' tool. If it is related to some other topic, use the respective tool. 
If no relevant tool applies, respond with "I don't know".

After determining the tool, **execute it using the provided input and include its output in your final response**.
"""
)

# agent_prompt = hub.pull("hwchase17/openai-functions-agent", api_key=HF_TOKEN )

agent = create_openai_functions_agent(
    llm=chat_model,
    prompt=agent_prompt,
    tools=tools,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=True,
)


if __name__ == '__main__':
    # Define an example question to test the agent
    # question = "What are the main uses of Amazon Sagemaker?"
    # question = "How to play Pokemon go?"
    question = "How to play Reading robot?"

    # Invoke the agent
    response = agent_executor.invoke({"question": question})
    # print(agent_executor.tool)

    # Print the response
    print(response)