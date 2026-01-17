print("Welcome in Vector Embeddings Document Part")

# Load Dataset - Process 1 : 
import pandas as pd
dataset = pd.read_csv("English_Dataset_Clean.csv")

# Import Necessary Module - Process 2 : 
import os
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Instatiate the Ollama Embeddings and Other Important Things - Process 3 : 
path_db = "./database_vec"
not_exists = not os.path.exists(path_db)
contents = [] # Akan berisi List of Document

try : 
    model_embeddings = OllamaEmbeddings(model = "mxbai-embed-large")
except : 
    print("HAVE SOME ERROR!!")
finally : 
    print("Model Embeddings Process Already Done")

if not_exists : # If the Path Doesn't Exists, So We Need to Add Those
    for answer,subject,body in zip(dataset["answer"],dataset["subject"], dataset["body"]) :
        isi_page_content = f"Question:{body} Answer:{answer}"
        isinya = Document(
            page_content = str(isi_page_content),
            metadata = {"subject":subject}
        )
        contents.append(isinya)

chroma = Chroma(
    collection_name = "TicketRAG",
    embedding_function=model_embeddings,
    persist_directory=path_db
)

if not_exists : 
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 512,
        chunk_overlap = 100
    )
    splitter_result = splitter.split_documents(contents)

    batch_size = 5461 # Maximum Batch Size Allowed by Chroma 

    for index in range(0,len(splitter_result),batch_size) : 
        chroma.add_documents(documents = splitter_result[index:index+batch_size])