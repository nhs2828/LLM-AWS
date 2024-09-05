import os
import boto3
import logging
from langchain.prompts import ChatPromptTemplate
from transformers import AutoTokenizer, GenerationConfig, BitsAndBytesConfig, AutoModelForCausalLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough,RunnableParallel, RunnableLambda
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import login
from flask import Flask, request, jsonify
from flask_cors import CORS
import json

# init logger
logger = logging.getLogger()
logger.setLevel("INFO")

# set up CONSTANT
BUCKET = os.environ.get('BUCKET_NAME')
EMBEDDING_MODEL_ID = os.environ.get('EMBEDDING_MODEL_ID')

# init s3
s3 = boto3.client('s3')
logger.info('S3 connected!')

def get_vectorstore():
    list_files = []
    objects = s3.list_objects_v2(Bucket=BUCKET, Prefix='vectorstore/')
    # get all the files in vectorstore in s3
    for obj in objects['Contents']:
        list_files.append(obj['Key'])
    # create a folder to where the files will be downloaded from s3
    os.makedirs("/vectorstore")
    # download the files (faiss and pkl) to the created folder
    for file in list_files:
        file_name = file.split('/')[-1]
        s3.download_file(
            Bucket = BUCKET,
            Key = file,
            Filename = f'/vectorstore/{file_name}'
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
    return FAISS.load_local('/vectorstore', embeddings, allow_dangerous_deserialization=True)

# init vectortore
vectorstore = get_vectorstore()
retriever = vectorstore.as_retriever()

# init Model
## get API tokens from AWS Secrets Manager
secret_manager = boto3.client('secretsmanager',
                              region_name = os.environ.get('AWS_REGION'))
hf_token_value_reponse = secret_manager.get_secret_value(
   SecretID = 'yoursecret_name_here'
)
hf_token = json.loads(hf_token_value_reponse['SecretString']['HF_API_KEY'])
login(token=hf_token)
## init model and tokenizers
bnb_config = BitsAndBytesConfig(
    load_in_4bit = True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype='float16',
    bnb_4bit_use_double_quant = False
)
llm = AutoModelForCausalLM.from_pretrained(
    'mistralai/Mistral-7B-Instruct-v0.3',
    quantization_config = bnb_config
)
tokenizer = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.3')


logger.info('Model initialized!')

# prompt
prompt_template = """Answer the question below using the context, if you do not know the answer, just say that you don't know:

Context: {context}

Question: {question}

Answer: """
prompt = ChatPromptTemplate.from_template(prompt_template)

def llm_reply(request):
    """
    Generate the texte with LLM
    """
    reponse = tokenizer.decode(
        llm.generate(
        input_ids = tokenizer(request, padding='max_length', truncation=True, return_tensors='pt')['input_ids'].to('cuda'),
        generation_config = GenerationConfig(
            max_new_tokens = 1000,
            temperature = 1.0
        )
    )[0],
    skip_special_tokens = True)
    return reponse

def format_docs(docs):
  """
  Extract the documents returned by retriever, concatenate them
  """
  return "\n\n".join(doc.page_content for doc in docs)

def extract_ans(x):
  """
  Extract only the answer, since LLM will fill the answer in the prompt
  """
  if 'Answer:' in x:
    return x.split('Answer:')[-1].replace("\n\n", "")
  return x

retrieval = RunnableParallel(
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
)

# define RAG chain
rag_chain = (
    retrieval
    | prompt
    | RunnableLambda(llm_reply)
    | StrOutputParser()
    | RunnableLambda(extract_ans)
)
app = Flask("RAG")
CORS(app)

@app.route('/rep', methods = ['POST'])
def func():
    # process user question
    user_question = request.get_json()
    # RAG chain
    reponse = rag_chain.invoke(user_question['question'])
    # prepare answer
    result = {
        'status': 200,
        'rep': reponse
    }
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9999)