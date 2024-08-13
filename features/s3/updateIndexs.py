import os
from transformers import AutoTokenizer
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from dotenv import load_dotenv
from huggingface_hub import login
from tools import *

MODEL_NAME = 'google/gemma-2-2b'

def main():
    # Load environment variables for API keys
    load_dotenv(os.path.join(os.getcwd(), '.env'))
    # Login Huggingface
    login(token = os.environ.get('HUGGINGFACEHUB_API_TOKEN'))
    # Load the PDFs as documents
    loader = PyPDFDirectoryLoader(PDF_FOLDER)
    docs = loader.load()

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Load InstructEmbedding
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    embedder_model = HuggingFaceInstructEmbeddings(
        model_name = MODEL_NAME,
        model_kwargs = model_kwargs,
        encode_kwargs = encode_kwargs
    )
    # Indexing
    vectorstore = FAISS.from_documents(documents = splits, embedding = embedder_model)
    vectorstore.save_local(VECTOR_LOCAL)
    print("DONE !")

if __name__ == '__main__':
    main()