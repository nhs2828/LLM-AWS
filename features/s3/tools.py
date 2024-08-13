import os
import boto3

VECTOR_LOCAL ='vectorstore'
CURRENT_DIR = os.getcwd()
PDF_FOLDER = 'pdfs'
BUCKET_LLM = 'llms3bucket'

def upload_file(s3, filePath, prefix=''):
    # extract file name
    filename = filePath.split('/')[-1]
    s3.upload_file(Filename=filePath,
                   Bucket = BUCKET_LLM,
                   Key = f'{prefix}{filename}')
    
def upload_folder(s3, folderName):
    for _, _, files in os.walk(os.path.join(CURRENT_DIR, folderName)):
        for file in files:
            filePath = os.path.join(CURRENT_DIR, VECTOR_LOCAL, file)
            upload_file(s3, filePath, folderName+'/')

def delete_file(s3, key):
    s3.delete_object(
        Bucket = BUCKET_LLM,
        Key = key
    )

def download_file(s3, filePath, key):
    s3.download_file(
        Bucket = BUCKET_LLM,
        Key = key,
        Filename = filePath
    )