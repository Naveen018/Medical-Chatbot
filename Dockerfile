FROM python:3.10-slim-buster

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

RUN pip uninstall -y pinecone-plugin-inference \
    && pip install --no-cache-dir --upgrade pinecone-client langchain_pinecone

EXPOSE 8090

CMD ["python3", "app.py"]