import os # dev
import json
import boto3
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain_aws.embeddings import BedrockEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from botocore.exceptions import ClientError
import pandas as pd
import time
import logging

EMBEDDING_MODEL_ID = os.environ.get('EMBEDDING_MODEL_ID')
doc_file = 'document.csv'
prefix_docs = 'docs/'
s3_vectorstore = 'vectorstore/'

# init logger
logger = logging.getLogger()
logger.setLevel("INFO")

# init s3
s3 = boto3.client('s3')
logger.info('S3 connected!')

BUCKET = os.environ.get('BUCKET_NAME')

# user function;
def s3_csv_to_pdf(key_value):
    try:
        s3 = boto3.client('s3')
        obj = s3.get_object(Bucket=BUCKET, Key=key_value)
        df = pd.read_csv(obj['Body'])
        return df
    except ClientError as ex:
        if ex.response['Error']['Code'] == 'NoSuchKey':
            print("Key doesn't match. Please check the key value")

def vectorstore_to_df(vectorstore):
    data = []
    for k, v in vectorstore.docstore._dict.items():
        doc_name = v.metadata['source'].split('/')[-1]
        doc_content = v.page_content
        data.append({
            "chunk_id": k,
            "document": doc_name,
            "content": doc_content
        })
    df = pd.DataFrame(data)
    return df

def get_vectorstore():
    """
    Get the vectorstore and the embedding model from s3
    """
    list_files = []
    objects = s3.list_objects_v2(Bucket=BUCKET, Prefix=s3_vectorstore)
    # get all the files in vectorstore in s3
    for obj in objects['Contents']:
        list_files.append(obj['Key'])
    # create a folder to where the files will be downloaded from s3
    os.makedirs("/tmp/vectorstore")
    # download the files (faiss and pkl) to the created folder
    for file in list_files:
        file_name = file.split('/')[-1]
        s3.download_file(
            Bucket = BUCKET,
            Key = file,
            Filename = f'/tmp/vectorstore/{file_name}'
        )
    # load emb_model
    bedrock_runtime = boto3.client(
        service_name="bedrock-runtime",
        region_name="eu-west-3",
    )

    embeddings = BedrockEmbeddings(
        model_id=EMBEDDING_MODEL_ID,
        client=bedrock_runtime,
        region_name="eu-west-3",
    )
    logger.info('Load embedding model done!')

    # load the vectorstore
    return FAISS.load_local('/tmp/vectorstore', embeddings, allow_dangerous_deserialization=True), embeddings

def check_update():
    """
    Get information about our data, determine which files are removed, which files are added
    """
    # get the Dataframe contains information about the documents
    data = s3_csv_to_pdf(doc_file)
    list_docs_last_update = data['docName'].tolist()
    # get names of all docs in database
    list_docs = []
    objects = s3.list_objects_v2(Bucket=BUCKET, Prefix=prefix_docs)
    for obj in objects['Contents']:
        list_docs.append(obj['Key'].split('/')[-1])
    # get docs added
    list_docs_added = [x for x in list_docs if x not in list_docs_last_update]
    # get docs removed
    list_docs_removed = [x for x in list_docs_last_update if x not in list_docs]
    return list_docs_removed, list_docs_added

def update_vectorstore(vectorstore, list_docs_removed, list_docs_added, embeddings):
    df_vectorstore = vectorstore_to_df(vectorstore)
    # delete removed docs from vectorstore
    for doc in list_docs_removed:
        chunk_id_list = df_vectorstore.loc[df_vectorstore['document']==doc]['chunk_id'].tolist()
        vectorstore.delete(chunk_id_list)

    # update new files
    ## get only the new files
    if list_docs_added != []:
        os.makedirs("/tmp/new_docs")
    for doc in list_docs_added:
        s3.download_file(
            Bucket = BUCKET,
            Key = f'{prefix_docs}{doc}',
            Filename = f'/tmp/new_docs/{doc}'
        )
    # Indexing process
    logger.info('Indexing!')
    loader = PyPDFDirectoryLoader('/tmp/new_docs')
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    index_creator = VectorstoreIndexCreator(
        vectorstore_cls=FAISS,
        embedding=embeddings,
        text_splitter = text_splitter
    )
    index_from_loader = index_creator.from_loaders([loader])

    # Merge with old vectorstore
    vectorstore.merge_from(index_from_loader.vectorstore)

    # update vectorstore
    logger.info('Sync vectorstore!')
    vectorstore.save_local('/tmp')
    s3.upload_file(
        "/tmp/index.faiss", BUCKET, f"{s3_vectorstore}/index.faiss"
    )
    s3.upload_file("/tmp/index.pkl", BUCKET, f"{s3_vectorstore}/index.pkl")
    logger.info('Sync done!')

    # update csv file
    logger.info('Updating documents information!')
    doc_df = s3_csv_to_pdf(doc_file)
    # filter out removed doc
    doc_df = doc_df[~doc_df['docName'].isin(list_docs_removed)]
    # add new doc info
    new_data = []
    for root, _, files in os.walk('/tmp/new_docs'):
        for file in files:
            if file.split('.')[-1] == 'pdf':
                filePath = f'{root}/{file}'
                # get time in seconds
                ti_c = os.path.getctime(filePath)
                # convert to timestamp ctime = create, getmtime = modified
                c_ti = time.ctime(ti_c)
                # convert to time structure
                t_obj = time.strptime(c_ti)
                # converto to ISO 8601 format
                T_stamp = time.strftime("%Y-%m-%d %H:%M:%S", t_obj)
                new_data.append({
                    'docName': file,
                    'createdTime': T_stamp
                })
    doc_df = doc_df._append(new_data)
    # save as csv
    doc_df.to_csv(f'/tmp/{doc_file}', index=False)
    # update s3 file (overwrite)
    s3.upload_file(Filename=f'/tmp/{doc_file}',
                   Bucket = BUCKET,
                   Key = doc_file)
    logger.info('Update done!')
    
def handler(event, context):
    logger.info('Checking for documents update!')
    list_docs_removed, list_docs_added = check_update()
    logger.info('Done!')
    # if there is no modification
    if list_docs_removed == [] and list_docs_added == []:
        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "statusCode": 200,
                    "message": "No update needed!"
                }
            )
        }
    # if there is modification
    vectorstore, hf_emb = get_vectorstore()
    update_vectorstore(vectorstore, list_docs_removed, list_docs_added, hf_emb)
    return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "statusCode": 200,
                    "message": "Update done!",
                    "docsRemoved": list_docs_removed,
                    "docsAdded": list_docs_added
                }
            )
        }
