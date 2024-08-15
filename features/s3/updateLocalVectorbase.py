import os
from transformers import AutoTokenizer
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from dotenv import load_dotenv
from huggingface_hub import login
from tools import *
import pickle

HF_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'

def main():
    # Load the PDFs as documents
    loader = PyPDFDirectoryLoader(PDF_FOLDER)
    docs = loader.load()

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Load embedding model
    path_hf_emb_model = os.path.join(os.getcwd(), 'emb_models', HF_MODEL_NAME, 'emb_model.pkl')
    if not os.path.isfile(path_hf_emb_model):
        print("Embedding model does not exist locally, creating model...")
        # Load environment variables for API keys
        load_dotenv(os.path.join(os.getcwd(), '.env'))
        # Login Huggingface
        login(token = os.environ.get('HUGGINGFACEHUB_API_TOKEN'))
        #Load InstructEmbedding
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        embedder_model = HuggingFaceInstructEmbeddings(
            model_name = HF_MODEL_NAME,
            model_kwargs = model_kwargs,
            encode_kwargs = encode_kwargs
        )
        # save hf model
        os.makedirs(os.path.join(
            os.getcwd(),
            'emb_models',
            HF_MODEL_NAME
        ))
        print("Saving emb model ...")
        with open(path_hf_emb_model, 'wb') as file:
            pickle.dump(embedder_model, file)
    else:
        print("Embedding model exists locally, getting model...")
        with open(path_hf_emb_model, 'rb') as file:
            embedder_model = pickle.load(file)
    # Indexing
    print("Embedding documents !")
    vectorstore = FAISS.from_documents(documents = splits, embedding = embedder_model)
    vectorstore.save_local(VECTOR_LOCAL)
    print("DONE !")

if __name__ == '__main__':
    main()