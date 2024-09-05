This is my RAG project on AWS eco-system

![RAG](https://github.com/user-attachments/assets/43a014c9-0e55-4726-b4cb-d77d9400d6a1)
Diagram of the architecture

# Context:
We would like to have a server that could perform Q-A enchanced by RAG framework. The FAISS vectorstore for RAG is updated automatically when there are changes in the documents database, in order to stay relevant.

# Demonstration

![demonstration_RAG](https://github.com/user-attachments/assets/1d5b5f20-a17d-4fbb-99c4-eb6dfbe10b9b)

# Overview
1. When users send questions, EC2 instance will generate the answers using RAG framework with the LLM model.
2. We could update the documents stored in S3 bucket. Cron job from EventBridge will trigger Lambda function to update the vectorstore automatically as scheduled, using embedding model from Bedrock.
3. EC2 instance, deployed using Docker to ensure consistent setup. This is where LLM generates answers for users's questions using RAG framework. The LLM is taken from HuggingFace, configured with quantization to reduce memory footprints.
4. FAISS for indexing task, because FAISS offers the possibility to modify (merge, delete, ..) vectorstore. Also the performance of FAISS is better than other algorithms I have tested
5. Documents used as context are PDFs, in particulary Machinea Learning papers 

# Specs:
1. LLM choice: `Mistral-7B-Instruct-v0.3`
  - I have tried different models, for me Mistral-7B performs well with different RAG techniques (multi-queries, fusion, ...)
  - I used the model from `transformers` to be able to have more controls on the model (quantization, generation, fine-tune, ...)
  - It is possible to call LLM using HuggingFace API to reduce setup time, but have less control, higher delay in gererating task, more traffic will go through Internet
2. Embedding model: `Cohere English`
  - I used this model from AWS Bedrock. Since the embedding task is performed in Lambda function, calling saved models or init from `HuggingFaceInstructEmbeddings` would consume lots of time (which is a billing factor of Lmabda function)
3. Both Lambda function and EC2 instance are dockerized to ensure consistent set up.
 - EC2 instance: `g4dn.xlarge` 4vCPU 16Gb Memory, I used quantization 4 bits to reduce memory footprints so this is enough for `Mistral-7B`

# Summerize the steps:
1. Set up S3 bucket with appropriate access policies
   - Documents
   - FAISS vectorstore
   - Saved LLM model or embedding model if we want to reduce setup time
3. Set up Lamda function with appropriate IAM role (Bedrock and S3)
   - Update FAISS vectorstore in S3 based on the update of Documents on S3
5. Set up Cron job to run Lambda function in a schedule
   - Daily, once a week, ...
6. Set up Secrets Manager for API keys
7. Set up EC2 instance with appropriate IAM role (S3, Bedrock)
   Get API keys to set up model
   Generate replies for question retrieved from front-end

# Possible Improvement
1. Auto scaling + Elastic Load Balancer to scale in and out based on demanded, reduce cost.
2. User chat history:
 - Enable chat flow with the history of chat (use Flan-T5 fine-tuned on Summerization task to summerize recent chat logs and put into prompt as another section)
 - Save user chat (need to ask for accord of users), for further trend, behavior, analysis (Kinesis, S3, Athena)


