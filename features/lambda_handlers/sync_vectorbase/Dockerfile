FROM public.ecr.aws/lambda/python:3.9-x86_64

ENV BUCKET_NAME llms3bucket
ENV EMBEDDING_MODEL_ID cohere.embed-english-v3
ENV AWS_LAMBDA_FUNCTION_TIMEOUT="300"
ENV AWS_LAMBDA_FUNCTION_MEMORY_SIZE=3000

COPY ./requirements.txt ${LAMBDA_TASK_ROOT}

RUN python3 -m pip install -r requirements.txt --target ${LAMBDA_TASK_ROOT}

COPY lambda_function.py ${LAMBDA_TASK_ROOT}/

CMD [ "lambda_function.handler" ]
