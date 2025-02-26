# Internal hosted LLM is used to create embedding 
# And have been used in langchain based chain creation as a LLM
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings

# Documentation: https://python.langchain.com/api_reference/ollama/embeddings/langchain_ollama.embeddings.OllamaEmbeddings.html

# Locally hosted LLM
ollama_emb = OllamaEmbeddings(
    model="llama3"
)

if __name__ == '__main__':
    # Example to generate embeddings with llm
    r1 = ollama_emb.embed_documents(
        [
            "Alpha is the first letter of Greek alphabet",
            "Beta is the second letter of Greek alphabet",
        ]
    )
    print("Embedding vector: ", r1)
    
    r2 = ollama_emb.embed_query(
        "What is the second letter of Greek alphabet"
    )
    print("Embedding vector: ", r2)