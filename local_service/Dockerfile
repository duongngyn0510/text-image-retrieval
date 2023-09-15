FROM python:3.8

WORKDIR /app

COPY /app /app/app
COPY model/model_storage /app/model/model_storage
COPY database/.env /app/database/.env

WORKDIR /app/app

RUN pip install --upgrade pip && pip install -r requirements.txt --no-cache-dir
RUN pip install git+https://github.com/openai/CLIP.git

EXPOSE 30000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "30000"]

COPY API_KEY /app/API_KEY
