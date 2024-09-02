This is my project on AWS eco-system

![RAG](https://github.com/user-attachments/assets/5f456dd5-3053-48f4-a8e9-647d219d93f8)
Diagram of the architecture

# Context:
We would like to have a server that could perform Q-A enchanced by RAG framework. The FAISS vectorstore for RAG is updated automatically when there are changes in the documents database, in order to stay relevant.

# Overview
1. When users send questions, EC2 instance will generate the answer using RAG framework with the LLM model and vectorstore fetched from S3 Bucket.
2. We could update the documents stored in S3 bucket. EventBridge will trigger Lambda function to update the vectorstore automatically as scheduled, using embedding model from Bedrock.
3. EC2 instance, deployed using Docker to ensure consistent setup. This is where LLM generates answers for users's questions using RAG framework. The LLM is taken from HuggingFace, configured with quantization to reduce memory footprints, the API keys are stored in Secrets Manager.

