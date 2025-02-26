import os
import dotenv
import ast
from PyPDF2 import PdfReader
from typing_extensions import Concatenate
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma

from int_host_emd import ollama_emb
from utils import process_path


def create_persist_database(data_path, chroma_path, CHROMA_DIR):
    # if not os.access(data_path, os.R_OK):
    #     print(f"Cannot read the file: {data_path}")
    pdfreader = PdfReader(data_path)
    print(f"Number of pages: {len(pdfreader.pages)}")
    # read text from pdf
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    print(len(raw_text))

    # We need to split the text using Character Text Split such that it sshould not increse token size
    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1300,
        chunk_overlap  = 300,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)
    print("No. of split texts or chunks: ", len(texts))

    if len(texts) == 0:
        print("Not find any chunks from a documents")
        return

    # Create a persistent vector db
    # If number of chunks are less than 300, we store all chunks from a file
    if len(texts) > 300:
        print("Debug: ", os.listdir(CHROMA_DIR))
        if chroma_path.split('\\')[-1] in os.listdir(CHROMA_DIR):
            # Load the existing db
            print(f"Loading existing chroma db: {chroma_path}")
            db = Chroma(persist_directory=chroma_path, embedding_function=ollama_emb)
            # Print the current chunks in chroma db
            print("No. of current chunks: ", len(db.get()['ids'])) 
            db.add_texts(texts=texts)
        else:
            Chroma.from_texts(texts=texts[:300], embedding=ollama_emb, persist_directory=chroma_path)
    else:
        print("Debug: ", os.listdir(CHROMA_DIR))
        if chroma_path.split('\\')[-1] in os.listdir(CHROMA_DIR):
            # Load the existing db
            print(f"Loading existing chroma db: {chroma_path}")
            db = Chroma(persist_directory=chroma_path, embedding_function=ollama_emb)
            # Print the current chunks in chroma db
            print("No. of current chunks: ", len(db.get()['ids'])) 
            db.add_texts(texts=texts)
        else:
            Chroma.from_texts(texts=texts, embedding=ollama_emb, persist_directory=chroma_path)

dotenv.load_dotenv()

DATA = os.getenv('DATA')
CHROMA_DIR = os.getenv('CHROMA_DIR')
REVIEWS_CHROMA_PATHS = os.getenv('REVIEWS_CHROMA_PATHS')

# DATA and REVIEWS_CHROMA_PATHS are list type we need to use ast
if DATA:
    DATA = ast.literal_eval(DATA)
if REVIEWS_CHROMA_PATHS:
    REVIEWS_CHROMA_PATHS = ast.literal_eval(REVIEWS_CHROMA_PATHS)    

path = DATA[-1]
if process_path(path) == True:
    create_persist_database(path, REVIEWS_CHROMA_PATHS[-1], CHROMA_DIR)
elif process_path(path) == "dir":
    # Iterate over all files in the directory
    print("Dir dir dir ")
    for root, _, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            print(file_path)
            if file.lower().endswith('.pdf'):
                print(f"{file_path} ---- Processing...")
                create_persist_database(file_path, REVIEWS_CHROMA_PATHS[-1], CHROMA_DIR)
            else:
                print(f"Found non-PDF file: {file_path}. Skipping...")