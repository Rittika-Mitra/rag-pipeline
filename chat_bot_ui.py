import os
import requests
import streamlit as st

from agent import agent_executor
# from chatbot import format_docs, sagemaker_retriever, sagemaker_chain, games_retriever, games_chain
from chatbot import format_docs, sagemaker_chain, retrieve_by_vector, sagemaker_vector_db
from int_host_emd import ollama_emb

CHATBOT_URL = os.getenv("CHATBOT_URL", "http://localhost:8000/poc-rag-agent")

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This chatbot interfaces with a
        [LangChain](https://python.langchain.com/docs/get_started/introduction)
        agent designed to answer questions about the Amazon Sagemaker and Snowflake.
        The agent uses  retrieval-augment generation (RAG) over both
        structured and unstructured data that has been synthetically generated.
        """
    )

    expert_behv = st.selectbox("Select an expert behaviour:", ["Agent", "Sagemaker"])

    st.header("Example Questions")
    st.markdown("- What is the use of Amazon Sagemaker?")
    st.markdown("- How to make Snowflake connection ?")
    st.markdown(
        "- Who is the president of India?"
    )
    st.markdown("- What ingredients and tools do I need to bake a chocolate cake?")
    

st.title("QA Chatbot - Int Hosted LLM")
st.info(
    "Ask me questions about Amazon Sagemaker"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "output" in message.keys():
            st.markdown(message["output"])

        if "explanation" in message.keys():
            with st.status("How was this generated", state="complete"):
                st.info(message["explanation"])

if prompt := st.chat_input("What do you want to know?"):
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({"role": "user", "output": prompt})
    print("Debug prompt: ", prompt)

    data = {"text": prompt}

    with st.spinner("Searching for an answer..."):
        if expert_behv == "Agent":
            response = agent_executor.invoke({"question": data})

            if response:
                output_text = response["output"]
                explanation = response["intermediate_steps"]
            else:
                output_text = """An error occurred while processing your message.
                Please try again or rephrase your message."""
                explanation = output_text
        elif expert_behv == "Sagemaker":
            response = sagemaker_chain.invoke(prompt)

            if response:
                output_text = response
                explanation = format_docs(retrieve_by_vector(sagemaker_vector_db, ollama_emb, prompt))
            else:
                output_text = """An error occurred while processing your message.
                Please try again or rephrase your message."""
                explanation = output_text

    st.chat_message("assistant").markdown(output_text)
    st.status("How was this generated", state="complete").info(explanation)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "output": output_text,
            "explanation": explanation,
        }
    )