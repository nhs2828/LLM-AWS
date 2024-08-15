import os
import boto3
from updateLocalVectorbase import HF_MODEL_NAME
from tools import *

def main():
    s3 = boto3.client('s3')
    # Upload vectorstore
    print("Uploading vectorstore...")
    upload_folder(s3, VECTOR_LOCAL)
    # Upload embedding model
    print("Uploading embedding model...")
    emb_model_path = os.path.join(
        'emb_models',
        HF_MODEL_NAME,
        'emb_model.pkl'
    )
    upload_file(s3, emb_model_path, prefix=f"emb_models/{HF_MODEL_NAME}/")
    
if __name__ == '__main__':
    main()