This is my project on AWS eco-system

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

