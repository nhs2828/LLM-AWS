FROM python:3.9

ENV EMBEDDING_MODEL_ID cohere.embed-english-v3
ENV BUCKET_NAME llms3bucket
ENV AWS_REGION eu-west-3

COPY ./requirements.txt ./
RUN python3 -m pip install -r requirements.txt

EXPOSE 9999

COPY myapp.py ./

CMD ["gunicorn", "-b", "0.0.0.0:9999", "myapp:app"]