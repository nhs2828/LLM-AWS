import os
from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_aws.embeddings import BedrockEmbeddings
from dotenv import load_dotenv
from huggingface_hub import login
from tools import *
import pickle

HF_MODEL_NAME = 'sentence-transformers/all-mpnet-base-v2'
EMBEDDING_MODEL_ID = 'cohere.embed-english-v3'

def main():
    # Load the PDFs as documents
    loader = PyPDFDirectoryLoader(DOCS_FOLDER)
    #docs = loader.load()

    # Chunking
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
    #splits = text_splitter.split_documents(docs)

    # Load embedding model
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name="eu-west-3",
    )

    embeddings = BedrockEmbeddings(
        model_id=EMBEDDING_MODEL_ID,
        client=bedrock_runtime,
        region_name="eu-west-3",
    )

    # Indexing
    index_creator = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,
        embedding=embeddings,
        text_splitter = text_splitter
    )
    print("Embedding documents !")
    index_from_loader = index_creator.from_loaders([loader])
    #vectorstore = FAISS.from_documents(documents = splits, embedding = embedder_model)
    index_from_loader.vectorstore.save_local(VECTOR_LOCAL)
    print("DONE !")

if __name__ == '__main__':
    main()