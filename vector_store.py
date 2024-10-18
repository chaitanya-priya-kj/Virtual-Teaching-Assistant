from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
import os
import warnings
from langchain_community.embeddings import OpenAIEmbeddings

# Suppress LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=DeprecationWarning)

api_key = "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"  # Replace with your OpenAI API key
os.environ["OPENAI_API_KEY"] = api_key

def get_text_chunks(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunk = text[i:i+chunk_size]
        chunks.append(chunk)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_text_chunks_from_files(folder_path, chunk_size=500):
    text_chunks = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                chunks = get_text_chunks(text, chunk_size)
                text_chunks.extend(chunks)
    return text_chunks

def get_vectorstore_from_folder(folder_path):
    text_chunks = get_text_chunks_from_files(folder_path)
    vectorstore = get_vectorstore(text_chunks)
    return vectorstore

folder_path = 'study_material'
vectorstore = get_vectorstore_from_folder(folder_path)