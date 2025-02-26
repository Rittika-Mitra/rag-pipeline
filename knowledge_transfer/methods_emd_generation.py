# Method 1 - Without a need of Hugging face token - working fine
# This will not download model file to local 
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
embeddings = HuggingFaceEndpointEmbeddings()

sentence = "Hello when do you think a major epedemic like corona will come again"

query_result = embeddings.embed_query(sentence)
print(len(query_result))


#**********************************************************************#
## Method 2 - passing a HF token to access the Model
# They doesn't download model file to local 
# Can pass parameter api_url for other hosted endpoint
# https://python.langchain.com/api_reference/community/embeddings/langchain_community.embeddings.huggingface.HuggingFaceInferenceAPIEmbeddings.html

import getpass
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

inference_api_key = getpass.getpass("Enter your HF Inference API Key:\n\n")

embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=inference_api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
)

text = "This is a test document."
query_result = embeddings.embed_query(text)
print(query_result)

#*****************************************************************#
# Method 3-  with Ollama embeddings
## https://python.langchain.com/api_reference/ollama/embeddings/langchain_ollama.embeddings.OllamaEmbeddings.html#langchain_ollama.embeddings.OllamaEmbeddings
# We need to host a local ollama on system to work with Ollama embeddings
# This will require GPU on system to work with selected LLM via a Ollama locally

from langchain_ollama import OllamaEmbeddings

embeddings_model = OllamaEmbeddings(model="llama3")
embeddings = embeddings_model.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]
)
len(embeddings), len(embeddings[0])

# To query single piece of line
embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
embedded_query[:5]


