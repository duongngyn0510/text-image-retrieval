FROM python:3.8


WORKDIR /app

COPY /model_serving/app /app/model_serving/app

# Set PINECONE_APIKEY env variable in Jenkins dashboard
RUN echo $PINECONE_APIKEY > /app/model_serving/.env

WORKDIR /app/model_serving/app

RUN pip install --upgrade pip && pip install -r requirements.txt --no-cache-dir
RUN pip install git+https://github.com/openai/CLIP.git

EXPOSE 30000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "30000"]
