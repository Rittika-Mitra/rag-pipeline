# Reference 
# Refer below link for various end points by ollama for hosted LLM
# https://github.com/ollama/ollama/blob/main/docs/api.md

# To make a post request with hosted LLM
import requests

headers = {
    "Content-Type": "application/json",
}

url = "https://ollama.aes.zdidata.com/api/generate"

payload = {
        "model": "llama3.2:3b",
        "prompt": "Who is  a president of India, give a brief intro about him",
        "stream": False,    # To get a final response rather stream of response  
    }
response = requests.post(url, json=payload, headers=headers)
# print(response.text)
print(response.json())


#--------------------------------------------------------------------------------------------#

# POST request to hosted llm to get embedding
# https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
import requests

headers = {
    "Content-Type": "application/json",
}

url = "https://ollama.aes.zdidata.com/api/embed"

payload = {
        "model": "llama3.2:3b",
        "input": "Why is the sky blue?"
    }
response = requests.post(url, json=payload, headers=headers)
# print(response.json()['embeddings'])
print(len(response.json()['embeddings'][0]))

