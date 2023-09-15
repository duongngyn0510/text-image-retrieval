FROM python:3.8


WORKDIR /app
ARG PINECONE_APIKEY

COPY /model_serving/app /app/model_serving/app


WORKDIR /app/model_serving/app

RUN pip install --upgrade pip && pip install -r requirements.txt --no-cache-dir
RUN pip install git+https://github.com/openai/CLIP.git

EXPOSE 30000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "30000"]
