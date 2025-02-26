import dotenv
import os
import ast
# from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
# from langchain.schema.messages import SystemMessage
from int_host_emd import ollama_emb
# To create chat model
from langchain_ollama import ChatOllama

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema.runnable import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


dotenv.load_dotenv()
REVIEWS_CHROMA_PATHS = os.getenv('REVIEWS_CHROMA_PATHS')
if REVIEWS_CHROMA_PATHS:
    REVIEWS_CHROMA_PATHS = ast.literal_eval(REVIEWS_CHROMA_PATHS)

def retrieve_by_vector(vector_db, embedding_fn, query):
    print(query)
    query_vector = embedding_fn.embed_query(query)  # Generate query embedding
    docs = vector_db.similarity_search_by_vector(query_vector)  # Perform similarity search
    return docs

def format_docs(docs):
    context = "\n\n".join(doc.page_content for doc in docs)
    print(context)
    return context

# Locally hosted LLM
chat_model = ChatOllama(
    model = "llama3"
)


# # Example usage
# messages = [
#     ("system", "You are a helpful translator. Translate the user sentence to French."),
#     ("human", "I love programming."),
# ]
# chat_model.invoke(messages)

sagemaker_template_str = """
You're an assistant knowledgeable about Sagemaker. Your job is to answer Sagemaker-related questions **only** based on the below context. 
Do not answer if the question is not about Sagemaker or if the context provided does not contain relevant information. 
If you cannot find the answer in the context below, say: "I don't know."

context={context}
"""

# system_message = SystemMessage(
#     content="""You're an assistant knowledgeable about
#     healthcare. Only answer healthcare-related questions."""
# )

sagemaker_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"], template=sagemaker_template_str
    )
)

sagemaker_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["question"], template="{question}")
)
messages = [sagemaker_system_prompt, sagemaker_human_prompt]

sagemaker_prompt_template = ChatPromptTemplate(
    input_variables=["context", "question"], messages=messages
)

output_parser = StrOutputParser()

# Vector DB use embedding of a internal hosted LLM to create emb vector from a text 
# Load the created vector db from pdf for a sagemaker use case
sagemaker_vector_db = Chroma(
    persist_directory=REVIEWS_CHROMA_PATHS[0],
    embedding_function=ollama_emb,
)

# sagemaker_retriever = sagemaker_vector_db.as_retriever(search_type="similarity", k=5)
# sagemaker_retriever = sagemaker_vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

## Method 1 - retrieve vector db data based on query
# retrieved_documents = sagemaker_retriever.invoke(question)
# print(retrieved_documents)
# for i, document in enumerate(retrieved_documents):
#     print(f"Document {i+1}: {document.page_content}")

## Method 2 - retrieve vector db data based on query
# docs = sagemaker_vector_db.similarity_search(question)
# for i in range(len(docs)):
#     print(f"Document {i+1}: {docs[i].page_content}")

# sagemaker_chain = (
#     {"context": sagemaker_retriever | format_docs, "question": RunnablePassthrough()}
#     | sagemaker_prompt_template
#     | chat_model
#     | StrOutputParser()
# )

# To search with embeddings
sagemaker_chain = (
    {
        "context": lambda inputs: format_docs(
            retrieve_by_vector(sagemaker_vector_db, ollama_emb, inputs)
        ),
        "question": RunnablePassthrough(),
    }
    | sagemaker_prompt_template
    | chat_model
    | StrOutputParser()
)


question = """What is Sagemaker and 
what is the process of ML in sagemaker"""

# Example to generate response from sagemaker_chain
# print(sagemaker_chain.invoke(question))
